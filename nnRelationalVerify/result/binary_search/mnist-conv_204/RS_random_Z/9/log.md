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
execution time: IAR + LP analysis = 15.45 + 34.14 = 49.59 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -2.0738973, upper bound: 2.0738940


# Binary Search by BASE starts (time budget: 3550.41 seconds, max iter: 100)

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
Binary search time: 204.74 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.01953125


# Relational Split (RS_random_Z) starts
Time budget: 3345.67 seconds

## Binary search (step 0) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 4608
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 4557
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 906

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 961

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8604407, upper bound: 1.8365446
time: 4.57 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8365448, upper bound: 1.8604410
time: 4.61 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 9.20 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 9.20
Output dim: 9, lower bound: -1.8604407, upper bound: 1.8365446
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 9.20
Output dim: 9, lower bound: -1.8365448, upper bound: 1.8604410

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -4.1575813, 4.1672077
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.7881203, 3.7881203
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.6598902, 3.6541209
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.6346583, 3.6291060
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.3806114, 3.3865585
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -4.2221336, 4.2282796
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2089024
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 4608
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 4557
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 494

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 884

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8576810, upper bound: 1.8362746
time: 4.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8601705, upper bound: 1.8337850
time: 4.59 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -4.1672077, 4.1575813
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.7881203, 3.7881203
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.6541224, 3.6598892
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.6291060, 3.6346588
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.3865585, 3.3806109
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -4.2282801, 4.2221336
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2089024
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 4608
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 4557
type: RSZ, layer: 1, pos: 4610

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4597

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8353794, upper bound: 1.8604377
time: 4.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8365413, upper bound: 1.8592756
time: 4.49 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 23.88 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 23.88
Output dim: 9, lower bound: -1.8576810, upper bound: 1.8362746
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 23.88
Output dim: 9, lower bound: -1.8601705, upper bound: 1.8337850
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 23.88
Output dim: 9, lower bound: -1.8353794, upper bound: 1.8604377
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 23.88
Output dim: 9, lower bound: -1.8365413, upper bound: 1.8592756

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -4.1572990, 4.1660762
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.7881203, 3.7881203
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.6598883, 3.6546583
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.6325908, 3.6207037
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.3738718, 3.3848977
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -4.2184553, 4.2273712
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2089024
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 4557
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5875

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8576748, upper bound: 1.8276799
time: 4.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8252164, upper bound: 1.8277432
time: 5.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -4.1564503, 4.1669259
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.7881203, 3.7881203
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.6604271, 3.6541204
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.6262565, 3.6270375
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.3789511, 3.3798194
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -4.2212248, 4.2246008
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2089024
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4557
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 4608
type: RSZ, layer: 1, pos: 906

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4557

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8601032, upper bound: 1.8074662
time: 4.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8338787, upper bound: 1.8337184
time: 4.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -4.1615887, 4.1396456
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.7881203, 3.7881203
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.6314783, 3.6457348
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.6191387, 3.6142364
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.3734379, 3.3596449
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -4.2418594, 4.2319093
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2089024
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 4557
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 4608
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 5820

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5798

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8321603, upper bound: 1.8604347
time: 4.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8353766, upper bound: 1.8572185
time: 4.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -4.1492710, 4.1519618
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.7881203, 3.7881203
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.6399670, 3.6372461
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.6086845, 3.6246915
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.3655920, 3.3674908
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -4.2380552, 4.2357140
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2089024
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 4557
type: RSZ, layer: 1, pos: 5875

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5798

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8333221, upper bound: 1.8592729
time: 4.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8365384, upper bound: 1.8560567
time: 4.65 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 24.12 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.12
Output dim: 9, lower bound: -1.8576748, upper bound: 1.8276799
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.12
Output dim: 9, lower bound: -1.8252164, upper bound: 1.8277432
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.12
Output dim: 9, lower bound: -1.8601032, upper bound: 1.8074662
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.12
Output dim: 9, lower bound: -1.8338787, upper bound: 1.8337184
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.12
Output dim: 9, lower bound: -1.8321603, upper bound: 1.8604347
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.12
Output dim: 9, lower bound: -1.8353766, upper bound: 1.8572185
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.12
Output dim: 9, lower bound: -1.8333221, upper bound: 1.8592729
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.12
Output dim: 9, lower bound: -1.8365384, upper bound: 1.8560567

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -4.1489840, 4.1608787
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.7881203, 3.7881203
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.6650715, 3.6578927
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.6264338, 3.6108565
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.3655691, 3.3797069
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -4.2107201, 4.2225375
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2089024
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4557
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4557

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8576073, upper bound: 1.8013841
time: 4.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8313536, upper bound: 1.8276127
time: 4.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -4.1521025, 4.1577606
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.7881203, 3.7881203
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.6631241, 3.6598287
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.6227431, 3.6145506
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.3686810, 3.3765945
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -4.2136211, 4.2196364
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2089024
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 4608
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 4557
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 5820

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4610

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8252123, upper bound: 1.8268902
time: 4.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8243632, upper bound: 1.8277392
time: 4.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -4.1547098, 4.1676016
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.7881203, 3.7881203
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.6254292, 3.6322298
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.5937772, 3.6067162
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.3720236, 3.3687320
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -4.2341661, 4.2245026
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2089024
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4608
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 4597

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4608

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8583108, upper bound: 1.8074571
time: 4.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8600933, upper bound: 1.8056778
time: 4.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -4.1571255, 4.1651855
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.7881203, 3.7881203
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.6385365, 3.6191235
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.6059356, 3.5945587
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.3678637, 3.3728924
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -4.2211275, 4.2375412
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2089024
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 4608
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5820

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4597

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8327100, upper bound: 1.8337144
time: 5.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8338785, upper bound: 1.8325531
time: 4.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -4.1559649, 4.1306486
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.7881203, 3.7881203
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.6305008, 3.6441760
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.5913467, 3.5671473
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.3362985, 3.3380675
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -4.2246475, 4.2232556
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2089024
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 4557
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 5820

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 494

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8321475, upper bound: 1.8603525
time: 4.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8320853, upper bound: 1.8604232
time: 4.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -4.1525908, 4.1340227
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.7881203, 3.7881203
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.6299191, 3.6447577
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.5720501, 3.5864434
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.3518605, 3.3225060
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -4.2332058, 4.2146964
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2089024
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4557
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 4608
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 884

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4557

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8353093, upper bound: 1.8308580
time: 4.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8090735, upper bound: 1.8571507
time: 4.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -4.1436491, 4.1429648
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.7881203, 3.7881203
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.6389885, 3.6356874
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.5808907, 3.5776029
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.3284535, 3.3459134
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -4.2208424, 4.2270603
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2089024
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4557
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 4608
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 906

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4557

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8332546, upper bound: 1.8329668
time: 4.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8069642, upper bound: 1.8592057
time: 4.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -4.1402750, 4.1463389
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.7881203, 3.7881203
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.6384068, 3.6362691
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.5615950, 3.5968986
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.3440156, 3.3303518
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -4.2294016, 4.2185011
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2089024
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 4608
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 4557
type: RSZ, layer: 1, pos: 884

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4610

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8365342, upper bound: 1.8552039
time: 4.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8356851, upper bound: 1.8560523
time: 4.86 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 24.17 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.17
Output dim: 9, lower bound: -1.8576073, upper bound: 1.8013841
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.17
Output dim: 9, lower bound: -1.8313536, upper bound: 1.8276127
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.17
Output dim: 9, lower bound: -1.8252123, upper bound: 1.8268902
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.17
Output dim: 9, lower bound: -1.8243632, upper bound: 1.8277392
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.17
Output dim: 9, lower bound: -1.8583108, upper bound: 1.8074571
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.17
Output dim: 9, lower bound: -1.8600933, upper bound: 1.8056778
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.17
Output dim: 9, lower bound: -1.8327100, upper bound: 1.8337144
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.17
Output dim: 9, lower bound: -1.8338785, upper bound: 1.8325531
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.17
Output dim: 9, lower bound: -1.8321475, upper bound: 1.8603525
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.17
Output dim: 9, lower bound: -1.8320853, upper bound: 1.8604232
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.17
Output dim: 9, lower bound: -1.8353093, upper bound: 1.8308580
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.17
Output dim: 9, lower bound: -1.8090735, upper bound: 1.8571507
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.17
Output dim: 9, lower bound: -1.8332546, upper bound: 1.8329668
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.17
Output dim: 9, lower bound: -1.8069642, upper bound: 1.8592057
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.17
Output dim: 9, lower bound: -1.8365342, upper bound: 1.8552039
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.17
Output dim: 9, lower bound: -1.8356851, upper bound: 1.8560523

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -4.1472445, 4.1615543
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.7881203, 3.7881203
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.6300726, 3.6359997
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.5939565, 3.5905366
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.3586407, 3.3686190
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -4.2236605, 4.2224398
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2089024
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 4608
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 4597

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 906

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8576043, upper bound: 1.7947826
time: 4.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8509353, upper bound: 1.8013807
time: 4.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -4.1496601, 4.1591382
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.7881203, 3.7881203
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.6431789, 3.6228929
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.6061149, 3.5783792
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.3544807, 3.3727789
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -4.2106228, 4.2354784
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2089024
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 4608
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5820

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5798

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8302298, upper bound: 1.8276104
time: 4.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8302331, upper bound: 1.8240173
time: 4.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -4.1498060, 4.1452031
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.7881203, 3.7881203
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.6627054, 3.6575723
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.6225581, 3.6136265
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.3664570, 3.3645930
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -4.2131538, 4.2170792
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2089024
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4608
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 4557
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 5820

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4608

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8234043, upper bound: 1.8268805
time: 4.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8252028, upper bound: 1.8251000
time: 4.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -4.1395445, 4.1554646
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.7881203, 3.7881203
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.6608667, 3.6594110
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.6218190, 3.6143656
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.3566790, 3.3743711
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -4.2110634, 4.2191691
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2089024
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 4608
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 4557
type: RSZ, layer: 1, pos: 906

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5820

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8243587, upper bound: 1.8208472
time: 4.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8174720, upper bound: 1.8277344
time: 4.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -4.1516504, 4.1690106
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.7881203, 3.7881203
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.6263771, 3.6301894
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.5979328, 3.5976911
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.3671904, 3.3709364
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -4.2321148, 4.2254419
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2089024
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 4610

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5875

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8583046, upper bound: 1.7988536
time: 4.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8258462, upper bound: 1.7989169
time: 4.49 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -4.1547098, 4.1645427
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.7881203, 3.7881203
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.6233883, 3.6322298
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.5847521, 3.6067162
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.3720236, 3.3638992
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -4.2341661, 4.2224512
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2089024
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 906

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4597

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8589281, upper bound: 1.8056779
time: 4.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8600898, upper bound: 1.8045097
time: 4.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -4.1515026, 4.1472454
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.7881203, 3.7881203
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.6158934, 3.6049681
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.5959682, 3.5741363
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.3547421, 3.3519254
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -4.2347088, 4.2473183
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2089024
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 4608
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 494

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5875

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8327038, upper bound: 1.8251198
time: 4.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8002418, upper bound: 1.8251831
time: 4.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -4.1391859, 4.1595621
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.7881203, 3.7881203
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.6243811, 3.5964799
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.5855131, 3.5845919
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.3468971, 3.3597713
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -4.2309046, 4.2511225
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2089024
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 4608
type: RSZ, layer: 1, pos: 906

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4610

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8338741, upper bound: 1.8317000
time: 4.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8330209, upper bound: 1.8325487
time: 4.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -4.1555653, 4.1383505
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.7881203, 3.7881203
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.6397905, 3.6436920
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.5909433, 3.5748787
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.3535538, 3.3371692
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -4.2237387, 4.2407155
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2089024
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4557
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 4608
type: RSZ, layer: 1, pos: 5820

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4557

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8320802, upper bound: 1.8340538
time: 4.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8057844, upper bound: 1.8602848
time: 5.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -4.1559649, 4.1302485
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.7881203, 3.7881203
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.6300173, 3.6441760
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.5913467, 3.5667448
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.3354015, 3.3380675
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -4.2246475, 4.2223477
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2089024
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 4608
type: RSZ, layer: 1, pos: 4557
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 5820

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 884

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8314294, upper bound: 1.8601527
time: 4.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8314307, upper bound: 1.8565612
time: 4.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -4.1508493, 4.1346970
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.7881203, 3.7881203
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.5949211, 3.6228662
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.5395718, 3.5661235
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.3449335, 3.3114185
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -4.2461462, 4.2145977
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2089024
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4608
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4608

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8335291, upper bound: 1.8308474
time: 4.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8353000, upper bound: 1.8290498
time: 4.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -4.1532660, 4.1322808
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.7881203, 3.7881203
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.6080284, 3.6097593
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.5517302, 3.5539651
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.3407736, 3.3155785
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -4.2331076, 4.2276363
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2089024
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 4608
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 884

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5820

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8090681, upper bound: 1.8502600
time: 4.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8021994, upper bound: 1.8571464
time: 4.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -4.1419077, 4.1436396
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.7881203, 3.7881203
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.6039906, 3.6137958
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.5484123, 3.5572824
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.3215256, 3.3348260
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -4.2337818, 4.2269616
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2089024
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 906

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 884

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8326063, upper bound: 1.8327074
time: 4.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8326076, upper bound: 1.8290726
time: 4.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -4.1443233, 4.1412234
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.7881203, 3.7881203
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.6170979, 3.6006889
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.5605707, 3.5451245
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.3173656, 3.3389859
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -4.2207432, 4.2400002
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2089024
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608
type: RSZ, layer: 1, pos: 906

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5820

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8069590, upper bound: 1.8523146
time: 4.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8000931, upper bound: 1.8592008
time: 4.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -4.1379728, 4.1337757
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.7881203, 3.7881203
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.6379910, 3.6340137
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.5614100, 3.5959749
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.3417912, 3.3183508
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -4.2289343, 4.2159433
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2089024
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 4608
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 4557

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 884

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8326723, upper bound: 1.8545560
time: 4.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8362640, upper bound: 1.8545544
time: 4.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -4.1277113, 4.1440411
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.7881203, 3.7881203
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.6361523, 3.6358528
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.5606699, 3.5967078
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.3320141, 3.3281312
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -4.2268438, 4.2180314
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2089024
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 4557
type: RSZ, layer: 1, pos: 4608
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 906

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 884

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8318232, upper bound: 1.8554045
time: 4.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8354149, upper bound: 1.8554027
time: 4.87 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 24.31 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 9, lower bound: -1.8576043, upper bound: 1.7947826
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 9, lower bound: -1.8509353, upper bound: 1.8013807
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 9, lower bound: -1.8302298, upper bound: 1.8276104
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 9, lower bound: -1.8302331, upper bound: 1.8240173
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 9, lower bound: -1.8234043, upper bound: 1.8268805
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 9, lower bound: -1.8252028, upper bound: 1.8251000
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 9, lower bound: -1.8243587, upper bound: 1.8208472
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 9, lower bound: -1.8174720, upper bound: 1.8277344
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 9, lower bound: -1.8583046, upper bound: 1.7988536
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 9, lower bound: -1.8258462, upper bound: 1.7989169
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 9, lower bound: -1.8589281, upper bound: 1.8056779
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 9, lower bound: -1.8600898, upper bound: 1.8045097
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 9, lower bound: -1.8327038, upper bound: 1.8251198
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 9, lower bound: -1.8002418, upper bound: 1.8251831
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 9, lower bound: -1.8338741, upper bound: 1.8317000
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 9, lower bound: -1.8330209, upper bound: 1.8325487
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 9, lower bound: -1.8320802, upper bound: 1.8340538
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 9, lower bound: -1.8057844, upper bound: 1.8602848
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 9, lower bound: -1.8314294, upper bound: 1.8601527
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 9, lower bound: -1.8314307, upper bound: 1.8565612
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 9, lower bound: -1.8335291, upper bound: 1.8308474
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 9, lower bound: -1.8353000, upper bound: 1.8290498
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 9, lower bound: -1.8090681, upper bound: 1.8502600
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 9, lower bound: -1.8021994, upper bound: 1.8571464
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 9, lower bound: -1.8326063, upper bound: 1.8327074
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 9, lower bound: -1.8326076, upper bound: 1.8290726
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 9, lower bound: -1.8069590, upper bound: 1.8523146
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 9, lower bound: -1.8000931, upper bound: 1.8592008
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 9, lower bound: -1.8326723, upper bound: 1.8545560
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 9, lower bound: -1.8362640, upper bound: 1.8545544
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 9, lower bound: -1.8318232, upper bound: 1.8554045
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 9, lower bound: -1.8354149, upper bound: 1.8554027

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -4.1439819, 4.1563473
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.7881203, 3.7881203
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.6202393, 3.6298475
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.5865335, 3.5786581
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.3524785, 3.3587661
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -4.2207613, 4.2206273
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2089024
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4608
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 5798

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4608

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8557966, upper bound: 1.7947702
time: 4.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8575948, upper bound: 1.7929863
time: 4.64 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 24.07 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 24.07
Output dim: 9, lower bound: -1.8557966, upper bound: 1.7947702
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 24.07
Output dim: 9, lower bound: -1.8575948, upper bound: 1.7929863
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.07
Output dim: 9, lower bound: -1.8509353, upper bound: 1.8013807
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.07
Output dim: 9, lower bound: -1.8302298, upper bound: 1.8276104
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.07
Output dim: 9, lower bound: -1.8302331, upper bound: 1.8240173
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.07
Output dim: 9, lower bound: -1.8234043, upper bound: 1.8268805
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.07
Output dim: 9, lower bound: -1.8252028, upper bound: 1.8251000
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.07
Output dim: 9, lower bound: -1.8243587, upper bound: 1.8208472
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.07
Output dim: 9, lower bound: -1.8174720, upper bound: 1.8277344
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.07
Output dim: 9, lower bound: -1.8583046, upper bound: 1.7988536
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.07
Output dim: 9, lower bound: -1.8258462, upper bound: 1.7989169
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.07
Output dim: 9, lower bound: -1.8589281, upper bound: 1.8056779
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.07
Output dim: 9, lower bound: -1.8600898, upper bound: 1.8045097
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.07
Output dim: 9, lower bound: -1.8327038, upper bound: 1.8251198
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.07
Output dim: 9, lower bound: -1.8002418, upper bound: 1.8251831
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.07
Output dim: 9, lower bound: -1.8338741, upper bound: 1.8317000
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.07
Output dim: 9, lower bound: -1.8330209, upper bound: 1.8325487
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.07
Output dim: 9, lower bound: -1.8320802, upper bound: 1.8340538
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.07
Output dim: 9, lower bound: -1.8057844, upper bound: 1.8602848
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.07
Output dim: 9, lower bound: -1.8314294, upper bound: 1.8601527
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.07
Output dim: 9, lower bound: -1.8314307, upper bound: 1.8565612
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.07
Output dim: 9, lower bound: -1.8335291, upper bound: 1.8308474
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.07
Output dim: 9, lower bound: -1.8353000, upper bound: 1.8290498
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.07
Output dim: 9, lower bound: -1.8090681, upper bound: 1.8502600
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.07
Output dim: 9, lower bound: -1.8021994, upper bound: 1.8571464
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.07
Output dim: 9, lower bound: -1.8326063, upper bound: 1.8327074
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.07
Output dim: 9, lower bound: -1.8326076, upper bound: 1.8290726
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.07
Output dim: 9, lower bound: -1.8069590, upper bound: 1.8523146
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.07
Output dim: 9, lower bound: -1.8000931, upper bound: 1.8592008
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.07
Output dim: 9, lower bound: -1.8326723, upper bound: 1.8545560
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.07
Output dim: 9, lower bound: -1.8362640, upper bound: 1.8545544
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.07
Output dim: 9, lower bound: -1.8318232, upper bound: 1.8554045
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.07
Output dim: 9, lower bound: -1.8354149, upper bound: 1.8554027
Binary search (step 0): status=Status.UNKNOWN, k_low=6, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=2.6592936515808105
rel_dist={9: [-1.860453671254131, 1.860453011622619]}

## Binary search (step 1) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4608
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4557
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4608

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7006541, upper bound: 1.7023512
time: 7.53 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7023515, upper bound: 1.7006542
time: 5.45 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 13.00 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 13.00
Output dim: 9, lower bound: -1.7006541, upper bound: 1.7023512
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 13.00
Output dim: 9, lower bound: -1.7023515, upper bound: 1.7006542

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.8834743, 3.8869495
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.6338367, 3.6261606
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.4629526, 3.4606276
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.4049182, 3.3946662
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.0994835, 3.1049571
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.9656734, 3.9679990
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2011461, 4.2016172
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4557
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 494

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 961

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7006441, upper bound: 1.6853641
time: 6.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6836645, upper bound: 1.7023417
time: 5.82 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.8865337, 3.8834743
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.6261606, 3.6329174
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.4606276, 3.4626679
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.3946662, 3.4036927
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.1043158, 3.0994835
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.9677248, 3.9656730
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2015610, 4.2011461
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4557
type: RSZ, layer: 1, pos: 4610

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4597

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7015348, upper bound: 1.7006523
time: 5.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7023490, upper bound: 1.6998336
time: 4.75 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 24.59 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.59
Output dim: 9, lower bound: -1.7006441, upper bound: 1.6853641
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.59
Output dim: 9, lower bound: -1.6836645, upper bound: 1.7023417
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.59
Output dim: 9, lower bound: -1.7015348, upper bound: 1.7006523
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.59
Output dim: 9, lower bound: -1.7023490, upper bound: 1.6998336

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.8732138, 3.8841758
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.6375093, 3.6261482
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.4674273, 3.4606161
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.4033232, 3.3887520
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.0931301, 3.1032295
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.9591236, 3.9662299
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2009420, 4.2008629
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4557
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 4597

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5875

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7006392, upper bound: 1.6769437
time: 7.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6752668, upper bound: 1.6769943
time: 4.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.8807001, 3.8766890
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.6338243, 3.6298332
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.4629402, 3.4651022
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.3990030, 3.3930707
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.0977554, 3.0986037
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.9639044, 3.9614496
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2003927, 4.2014122
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 4557
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 884

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4610

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6836610, upper bound: 1.7016643
time: 6.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6829876, upper bound: 1.7023384
time: 5.13 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.8781767, 3.8655386
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.6477013, 3.6487617
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.4379835, 3.4466276
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.3823748, 3.3832698
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.0894508, 3.0785170
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.9804602, 3.9754486
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2002115, 4.2073498
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 4557
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 5820

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5798

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6987693, upper bound: 1.7006514
time: 4.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7015323, upper bound: 1.6978736
time: 4.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.8685970, 3.8751178
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.6420059, 3.6544561
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.4445868, 3.4400253
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.3742437, 3.3914018
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.0833492, 3.0846195
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.9775000, 3.9784079
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2077646, 4.1997967
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 4557
type: RSZ, layer: 1, pos: 906

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5875

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7023446, upper bound: 1.6783677
time: 6.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6808828, upper bound: 1.6998294
time: 4.62 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 25.62 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.62
Output dim: 9, lower bound: -1.7006392, upper bound: 1.6769437
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.62
Output dim: 9, lower bound: -1.6752668, upper bound: 1.6769943
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.62
Output dim: 9, lower bound: -1.6836610, upper bound: 1.7016643
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.62
Output dim: 9, lower bound: -1.6829876, upper bound: 1.7023384
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.62
Output dim: 9, lower bound: -1.6987693, upper bound: 1.7006514
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.62
Output dim: 9, lower bound: -1.7015323, upper bound: 1.6978736
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.62
Output dim: 9, lower bound: -1.7023446, upper bound: 1.6783677
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.62
Output dim: 9, lower bound: -1.6808828, upper bound: 1.6998294

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.8648987, 3.8782854
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.6288385, 3.6200042
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.4721756, 3.4638491
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.3963461, 3.3789048
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.0848274, 3.0973463
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.9513884, 3.9607515
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2039337, 4.2052526
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 4557
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 494

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 906

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7006369, upper bound: 1.6719461
time: 6.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6956348, upper bound: 1.6769406
time: 5.49 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.8673229, 3.8758607
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.6263113, 3.6174769
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.4706612, 3.4653544
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.3934755, 3.3817778
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.0872478, 3.0949259
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.9536448, 3.9584951
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2042303, 4.2038546
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 4557
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 884

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 494

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6752563, upper bound: 1.6769107
time: 7.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6751840, upper bound: 1.6769836
time: 4.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.8761253, 3.8641319
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.6292095, 3.6171474
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.4621143, 3.4628458
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.3986540, 3.3921466
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.0933590, 3.0866022
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.9629717, 3.9588919
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2002182, 4.2009363
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 4557
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 5798

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 906

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6836587, upper bound: 1.6966725
time: 5.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6786555, upper bound: 1.7016616
time: 7.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.8681431, 3.8721132
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.6211395, 3.6252179
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.4606838, 3.4642763
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.3980799, 3.3927212
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.0857544, 3.0942068
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.9613457, 3.9605179
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.1999149, 4.2012396
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 4557
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 906

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5820

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6829811, upper bound: 1.6968004
time: 4.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6774474, upper bound: 1.7023318
time: 4.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.8718052, 3.8565416
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.6462517, 3.6452541
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.4368782, 3.4450684
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.3502932, 3.3361812
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.0523124, 3.0534816
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.9632473, 3.9648943
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.1845665, 4.1962681
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 4557
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 884

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 961

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6987593, upper bound: 1.6836619
time: 5.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6817807, upper bound: 1.7006414
time: 5.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.8691807, 3.8591661
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.6441917, 3.6473141
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.4364262, 3.4455209
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.3352852, 3.3511891
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.0644155, 3.0413780
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.9699049, 3.9582372
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.1891308, 4.1917038
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 4557
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 494

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5875

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7015279, upper bound: 1.6764095
time: 5.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6800669, upper bound: 1.6978709
time: 4.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.8602810, 3.8692269
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.6333361, 3.6483150
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.4493246, 3.4432573
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.3672705, 3.3815565
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.0750446, 3.0787358
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.9697657, 3.9729300
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2041845
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 4557

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4610

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7023411, upper bound: 1.6776916
time: 6.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7016670, upper bound: 1.6783642
time: 6.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.8627062, 3.8668017
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.6358633, 3.6457882
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.4478197, 3.4447627
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.3643980, 3.3844295
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.0774651, 3.0763159
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.9720221, 3.9706731
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2027864
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4557
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 494

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4610

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6808792, upper bound: 1.6991527
time: 7.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6802052, upper bound: 1.6998255
time: 4.62 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 26.26 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.26
Output dim: 9, lower bound: -1.7006369, upper bound: 1.6719461
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.26
Output dim: 9, lower bound: -1.6956348, upper bound: 1.6769406
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.26
Output dim: 9, lower bound: -1.6752563, upper bound: 1.6769107
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.26
Output dim: 9, lower bound: -1.6751840, upper bound: 1.6769836
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.26
Output dim: 9, lower bound: -1.6836587, upper bound: 1.6966725
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.26
Output dim: 9, lower bound: -1.6786555, upper bound: 1.7016616
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.26
Output dim: 9, lower bound: -1.6829811, upper bound: 1.6968004
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.26
Output dim: 9, lower bound: -1.6774474, upper bound: 1.7023318
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.26
Output dim: 9, lower bound: -1.6987593, upper bound: 1.6836619
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.26
Output dim: 9, lower bound: -1.6817807, upper bound: 1.7006414
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.26
Output dim: 9, lower bound: -1.7015279, upper bound: 1.6764095
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.26
Output dim: 9, lower bound: -1.6800669, upper bound: 1.6978709
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.26
Output dim: 9, lower bound: -1.7023411, upper bound: 1.6776916
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.26
Output dim: 9, lower bound: -1.7016670, upper bound: 1.6783642
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.26
Output dim: 9, lower bound: -1.6808792, upper bound: 1.6991527
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.26
Output dim: 9, lower bound: -1.6802052, upper bound: 1.6998255

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.8612061, 3.8730803
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.6251888, 3.6174202
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.4623413, 3.4568772
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.3879333, 3.3670268
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.0778446, 3.0874939
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.9484901, 3.9586983
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.1963854, 4.1946020
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4557

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4610

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7006334, upper bound: 1.6712626
time: 7.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6999596, upper bound: 1.6719402
time: 6.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.8596935, 3.8745933
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.6262541, 3.6163554
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.4652042, 3.4540148
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.3844676, 3.3704920
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.0749741, 3.0903640
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.9493351, 3.9578528
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.1932831, 4.1977043
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4557
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 5820

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4557

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6955801, upper bound: 1.6555040
time: 8.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6741892, upper bound: 1.6768863
time: 5.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.8669233, 3.8817616
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.6404324, 3.6165180
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.4777784, 3.4648709
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.3930721, 3.3877010
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.1004677, 3.0940266
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.9527359, 3.9718723
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2034683
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 4557
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 5798

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 884

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6731606, upper bound: 1.6766950
time: 7.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6750463, upper bound: 1.6748143
time: 4.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.8673229, 3.8754597
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.6253519, 3.6174769
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.4701757, 3.4653544
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.3934755, 3.3813748
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.0863476, 3.0949259
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.9536448, 3.9575858
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2038441, 4.2038546
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4557
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5820

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 884

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6730884, upper bound: 1.6767713
time: 4.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6749719, upper bound: 1.6748871
time: 5.04 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.8724308, 3.8589249
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.6255608, 3.6145639
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.4522810, 3.4558749
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.3902421, 3.3802686
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.0863762, 3.0767484
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.9600744, 3.9568400
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.1926727, 4.1902876
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 4557
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 5875

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 884

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6815629, upper bound: 1.6964362
time: 4.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6834371, upper bound: 1.6945772
time: 4.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.8709173, 3.8604379
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.6266260, 3.6134992
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.4551439, 3.4530120
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.3867764, 3.3837337
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.0835056, 3.0796185
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.9609203, 3.9559946
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.1895695, 4.1933899
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 4557
type: RSZ, layer: 1, pos: 884

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5820

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6779716, upper bound: 1.6959766
time: 4.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6779665, upper bound: 1.7016555
time: 5.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.8614225, 3.8626356
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.6138744, 3.6200695
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.4054375, 3.4280152
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.3817091, 3.3696132
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.0839162, 3.0916128
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.9481840, 3.9515157
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.1661091, 4.1535168
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4557
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 5798

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4557

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6829262, upper bound: 1.6753423
time: 4.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6615601, upper bound: 1.6967454
time: 5.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.8586664, 3.8653922
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.6159906, 3.6179533
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.4244232, 3.4090290
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.3749714, 3.3763514
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.0831609, 3.0923686
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.9523439, 3.9473562
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.1521931, 4.1674328
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4557
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 5875

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4597

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6766224, upper bound: 1.7023294
time: 4.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6774446, upper bound: 1.7015148
time: 4.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.8615427, 3.8537683
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.6499252, 3.6452408
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.4413509, 3.4450564
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.3486958, 3.3302670
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.0459580, 3.0517540
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.9566975, 3.9631238
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.1843634, 4.1955156
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 4557
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 5875

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 884

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6982192, upper bound: 1.6834385
time: 4.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6982205, upper bound: 1.6803502
time: 4.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.8690300, 3.8462815
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.6462402, 3.6489253
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.4368649, 3.4495430
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.3443775, 3.3345857
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.0505853, 3.0471282
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.9614773, 3.9583435
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.1838131, 4.1960649
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 4557
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 5875

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 884

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6812394, upper bound: 1.7004199
time: 4.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6812407, upper bound: 1.6973289
time: 5.25 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.8608656, 3.8532758
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.6355209, 3.6411691
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.4411650, 3.4487543
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.3283119, 3.3413429
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.0561128, 3.0354953
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.9621696, 3.9527588
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.1921225, 4.1960926
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 4557
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 961

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5820

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7015214, upper bound: 1.6708620
time: 4.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6959954, upper bound: 1.6764010
time: 5.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.8632908, 3.8508506
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.6380482, 3.6386418
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.4396582, 3.4502602
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.3254385, 3.3442159
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.0585332, 3.0330749
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.9644260, 3.9505024
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.1935196, 4.1946945
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4557
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 5820

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 961

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6761484, upper bound: 1.6725157
time: 4.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6761281, upper bound: 1.6978584
time: 4.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.8556976, 3.8566623
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.6287136, 3.6356220
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.4485006, 3.4410014
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.3669224, 3.3806329
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.0706501, 3.0667353
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.9688339, 3.9703722
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2037086
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 4557
type: RSZ, layer: 1, pos: 494

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5820

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7023346, upper bound: 1.6721465
time: 5.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6968033, upper bound: 1.6776857
time: 4.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.8477163, 3.8646464
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.6206436, 3.6436830
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.4470701, 3.4424319
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.3663464, 3.3812032
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.0630455, 3.0743423
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.9672079, 3.9719963
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2040119
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 4557
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 5798

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 961

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7016566, upper bound: 1.6744292
time: 4.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6763138, upper bound: 1.6744501
time: 4.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.8581228, 3.8542376
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.6312408, 3.6330948
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.4469948, 3.4425068
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.3640480, 3.3835058
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.0730705, 3.0643148
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.9710903, 3.9681158
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2023115
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.57 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=6, k_high=8, k_mid=7, eps_mid=0.0273438, abs_max=2.6592936515808105
rel_dist={9: [-1.702359651903449, 1.7023598330847598]}

## Binary search (step 2) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 4608
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 4557
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 4610

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5875

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6176209, upper bound: 1.5992303
time: 4.80 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5992299, upper bound: 1.6176213
time: 5.28 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 10.10 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 10.10
Output dim: 9, lower bound: -1.6176209, upper bound: 1.5992303
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 10.10
Output dim: 9, lower bound: -1.5992299, upper bound: 1.6176213

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.7375641, 3.7396431
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.4934311, 3.4955969
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.3714609, 3.3701696
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0438118, 5.0470133
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.2806449, 3.2781820
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -2.9546871, 2.9567609
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.8295107, 3.8314452
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.0943346, 4.0955315
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4557
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 4608
type: RSZ, layer: 1, pos: 5820

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4557

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6175706, upper bound: 1.5807222
time: 5.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5991128, upper bound: 1.5991798
time: 12.63 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.7396431, 3.7375641
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.4955969, 3.4934306
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.3701696, 3.3714604
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0470123, 5.0438108
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.2781825, 3.2806444
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -2.9567614, 2.9546866
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.8314457, 3.8295107
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.0955324, 4.0943336
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 4608
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 4557
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 884

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 961

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5958572, upper bound: 1.5958831
time: 5.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5958396, upper bound: 1.6176124
time: 4.95 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 24.95 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.95
Output dim: 9, lower bound: -1.6175706, upper bound: 1.5807222
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 24.95
Output dim: 9, lower bound: -1.5991128, upper bound: 1.5991798
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 24.95
Output dim: 9, lower bound: -1.5958572, upper bound: 1.5958831
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.95
Output dim: 9, lower bound: -1.5958396, upper bound: 1.6176124

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.7358227, 3.7395124
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.4912720, 3.4939756
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.3364639, 3.3439102
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0415325, 5.0453053
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.2481666, 3.2538099
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -2.9463720, 2.9456735
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.8381052, 3.8313465
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.0847836, 4.0827980
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 4608
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5820

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6175656, upper bound: 1.5759439
time: 4.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.6127912, upper bound: 1.5807187
time: 5.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.7357988, 3.7273026
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.4955845, 3.4965763
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.3701582, 3.3753033
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0448570, 5.0380411
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.2722669, 3.2784300
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -2.9543724, 2.9483328
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.8289928, 3.8229618
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.0947781, 4.0940514
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 4608
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4557
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 4597

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 494

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5958300, upper bound: 1.6175328
time: 5.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5957600, upper bound: 1.6176025
time: 6.37 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 25.93 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.93
Output dim: 9, lower bound: -1.6175656, upper bound: 1.5759439
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 25.93
Output dim: 9, lower bound: -1.6127912, upper bound: 1.5807187
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.93
Output dim: 9, lower bound: -1.5958300, upper bound: 1.6175328
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.93
Output dim: 9, lower bound: -1.5957600, upper bound: 1.6176025

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.7287083, 3.7300348
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.4840088, 3.4885273
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.2812138, 3.3049359
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0276909, 5.0209885
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.2308335, 3.2307010
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -2.9444265, 2.9430804
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.8249445, 3.8217506
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.0489912, 4.0350771
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.6963558, 3.7068610
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 4608
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 4610

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 906

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6175639, upper bound: 1.5757701
time: 4.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.6126238, upper bound: 1.5757718
time: 7.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.7353973, 3.7323031
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.5075512, 3.4956169
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.3761892, 3.3748178
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0640316, 5.0365038
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.2718644, 3.2834492
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -2.9655762, 2.9474335
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.8280840, 3.8342977
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.0995884, 4.0936651
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 4557
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 906

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5958280, upper bound: 1.6134328
time: 5.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5917301, upper bound: 1.6175306
time: 5.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.7357988, 3.7269020
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.4946251, 3.4965763
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.3696737, 3.3753033
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0433216, 5.0380411
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.2722669, 3.2780266
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -2.9534740, 2.9483328
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.8289928, 3.8220525
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.0943918, 4.0940514
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4608
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4557
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 906

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4608

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5940838, upper bound: 1.6175964
time: 5.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5957538, upper bound: 1.6159251
time: 4.78 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 24.48 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.48
Output dim: 9, lower bound: -1.6175639, upper bound: 1.5757701
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 24.48
Output dim: 9, lower bound: -1.6126238, upper bound: 1.5757718
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 24.48
Output dim: 9, lower bound: -1.5958280, upper bound: 1.6134328
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.48
Output dim: 9, lower bound: -1.5917301, upper bound: 1.6175306
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.48
Output dim: 9, lower bound: -1.5940838, upper bound: 1.6175964
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 24.48
Output dim: 9, lower bound: -1.5957538, upper bound: 1.6159251

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.7248001, 3.7248297
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.4803581, 3.4857893
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.2879543, 3.3141322
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0355673, 5.0267601
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.2219238, 3.2188220
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -2.9370346, 2.9332275
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.8249369, 3.8224688
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.0410013, 4.0244284
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.6905060, 3.7047615
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 4608
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 494

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 884

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.6157849, upper bound: 1.5756622
time: 5.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6174551, upper bound: 1.5739917
time: 5.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.7301922, 3.7283945
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.5048151, 3.4919682
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.3688087, 3.3649840
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0556202, 5.0301971
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.2599869, 3.2745419
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -2.9557238, 2.9400411
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.8259096, 3.8313990
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.0889368, 4.0856743
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 4557
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5798

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5892964, upper bound: 1.6175288
time: 4.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5917281, upper bound: 1.6150979
time: 5.27 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.7327404, 3.7268219
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.4944487, 3.4898200
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.3696260, 3.3732610
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0431499, 5.0380335
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.2720280, 3.2690001
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -2.9486427, 2.9481931
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.8269415, 3.8219943
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.0939770, 4.0940409
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 4557
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 884

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 906

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5940817, upper bound: 1.6134962
time: 5.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5899842, upper bound: 1.6175941
time: 4.82 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 24.79 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.79
Output dim: 9, lower bound: -1.6157849, upper bound: 1.5756622
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.79
Output dim: 9, lower bound: -1.6174551, upper bound: 1.5739917
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.79
Output dim: 9, lower bound: -1.5892964, upper bound: 1.6175288
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.79
Output dim: 9, lower bound: -1.5917281, upper bound: 1.6150979
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.79
Output dim: 9, lower bound: -1.5940817, upper bound: 1.6134962
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.79
Output dim: 9, lower bound: -1.5899842, upper bound: 1.6175941

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.7236691, 3.7242651
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.4799032, 3.4855499
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.2883129, 3.3141313
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0340929, 5.0237913
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.2135215, 3.2146416
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -2.9336815, 2.9264884
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.8231068, 3.8187919
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.0395765, 4.0215588
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.6905117, 3.7042017
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 961

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6174462, upper bound: 1.5706011
time: 5.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5957170, upper bound: 1.5706186
time: 5.08 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.7234468, 3.7193995
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.5030708, 3.4884577
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.3676376, 3.3634253
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0384626, 5.0173321
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.2257605, 3.2274518
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -2.9185858, 2.9132767
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.8086958, 3.8198919
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.0732937, 4.0739422
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4557
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 4608
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 5820

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4557

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5892461, upper bound: 1.5990215
time: 4.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5707845, upper bound: 1.6174782
time: 5.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.7275352, 3.7229123
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.4917107, 3.4861717
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.3622456, 3.3634267
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0347404, 5.0317278
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.2601504, 3.2600923
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -2.9387903, 2.9408007
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.8247671, 3.8190966
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.0833273, 4.0860491
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 4557
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 884

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5798

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5875475, upper bound: 1.6175922
time: 4.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5899821, upper bound: 1.6151612
time: 4.79 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 24.29 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 24.29
Output dim: 9, lower bound: -1.6174462, upper bound: 1.5706011
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 24.29
Output dim: 9, lower bound: -1.5957170, upper bound: 1.5706186
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 24.29
Output dim: 9, lower bound: -1.5892461, upper bound: 1.5990215
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 24.29
Output dim: 9, lower bound: -1.5707845, upper bound: 1.6174782
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 24.29
Output dim: 9, lower bound: -1.5875475, upper bound: 1.6175922
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 24.29
Output dim: 9, lower bound: -1.5899821, upper bound: 1.6151612

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.7134075, 3.7204208
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.4830494, 3.4855385
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.2921543, 3.3141184
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0283232, 5.0216370
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.2113066, 3.2087264
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -2.9273281, 2.9240999
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.8165579, 3.8163395
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.0392914, 4.0208035
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.6817961, 3.7009387
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4608
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 494

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4608

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.6157696, upper bound: 1.5705947
time: 5.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6174401, upper bound: 1.5689264
time: 5.13 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.7233143, 3.7176561
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.5014505, 3.4862995
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.3413801, 3.3284287
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0367527, 5.0150528
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.2013874, 3.1949739
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -2.9074979, 2.9049625
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.8085985, 3.8284864
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.0605583, 4.0643902
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 4608
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 5820

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4597

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5699903, upper bound: 1.6174765
time: 5.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5707822, upper bound: 1.6167054
time: 5.46 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.7207890, 3.7139177
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.4899664, 3.4826603
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.3610735, 3.3618679
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0175829, 5.0188618
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.2259221, 3.2130017
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -2.9016504, 2.9140363
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.8075533, 3.8075867
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.0676813, 4.0743170
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4557

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4610

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5875441, upper bound: 1.6170097
time: 5.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5869639, upper bound: 1.6175886
time: 5.13 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 24.81 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 24.81
Output dim: 9, lower bound: -1.6157696, upper bound: 1.5705947
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 24.81
Output dim: 9, lower bound: -1.6174401, upper bound: 1.5689264
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 24.81
Output dim: 9, lower bound: -1.5699903, upper bound: 1.6174765
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 24.81
Output dim: 9, lower bound: -1.5707822, upper bound: 1.6167054
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 24.81
Output dim: 9, lower bound: -1.5875441, upper bound: 1.6170097
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 24.81
Output dim: 9, lower bound: -1.5869639, upper bound: 1.6175886

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.7133274, 3.7173624
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.4762936, 3.4853592
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.2901115, 3.3140702
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0283155, 5.0214653
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.2022800, 3.2084928
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -2.9271889, 2.9192677
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.8164997, 3.8142877
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.0392780, 4.0203848
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.6805649, 3.7009087
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 494

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6174304, upper bound: 1.5688458
time: 5.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6173603, upper bound: 1.5689160
time: 4.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.7135878, 3.6997180
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.5221729, 3.5021467
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.3187361, 3.3114443
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0348701, 5.0136328
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.1879354, 3.1745515
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -2.8917608, 2.8839960
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.8209114, 3.8382630
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.0592098, 4.0695152
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5820

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5691427, upper bound: 1.6125281
time: 5.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5691373, upper bound: 1.6174716
time: 5.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.7150707, 3.7013588
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.4841967, 3.4699764
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.3600454, 3.3596129
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0078325, 5.0144386
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.2254910, 3.2120781
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -2.8961678, 2.9020348
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.8063898, 3.8050289
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.0674629, 4.0738392
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 4557
type: RSZ, layer: 1, pos: 4597

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 884

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5871356, upper bound: 1.6169009
time: 5.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5871367, upper bound: 1.6141691
time: 6.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.7082300, 3.7081995
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.4772797, 3.4768939
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.3588190, 3.3608389
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0131578, 5.0091133
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.2249990, 3.2125707
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -2.8896484, 2.9085531
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.8049965, 3.8064222
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.0672035, 4.0740995
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 4557
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 884

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5820

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5861178, upper bound: 1.6126397
time: 4.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5861123, upper bound: 1.6175836
time: 5.09 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 24.51 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 24.51
Output dim: 9, lower bound: -1.6174304, upper bound: 1.5688458
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 24.51
Output dim: 9, lower bound: -1.6173603, upper bound: 1.5689160
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 24.51
Output dim: 9, lower bound: -1.5691427, upper bound: 1.6125281
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 24.51
Output dim: 9, lower bound: -1.5691373, upper bound: 1.6174716
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 24.51
Output dim: 9, lower bound: -1.5871356, upper bound: 1.6169009
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 24.51
Output dim: 9, lower bound: -1.5871367, upper bound: 1.6141691
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 24.51
Output dim: 9, lower bound: -1.5861178, upper bound: 1.6126397
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 24.51
Output dim: 9, lower bound: -1.5861123, upper bound: 1.6175836

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.7129269, 3.7223630
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.4882612, 3.4844003
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.2961454, 3.3135872
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0474882, 5.0199299
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.2018766, 3.2135119
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -2.9383912, 2.9183669
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.8155899, 3.8256235
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.0440874, 4.0199986
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.6803474, 3.7036180
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4597

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.6166572, upper bound: 1.5688439
time: 5.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6174283, upper bound: 1.5680481
time: 5.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.7133274, 3.7169619
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.4753342, 3.4853592
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.2896299, 3.3140702
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0267801, 5.0214653
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.2022800, 3.2080894
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -2.9262881, 2.9192677
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.8164997, 3.8133779
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.0388918, 4.0203848
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.6805649, 3.7006903
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 4610

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5798

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.6146287, upper bound: 1.5678499
time: 5.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6173582, upper bound: 1.5678481
time: 4.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.7041111, 3.6926060
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.5167227, 3.4948826
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.2963390, 3.2727714
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0247297, 5.0139732
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.1648264, 3.1572170
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -2.8891683, 2.8820519
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.8142090, 3.8279948
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.0114899, 4.0337238
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7141762, 3.7019744
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4610

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5691333, upper bound: 1.6168884
time: 4.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5685537, upper bound: 1.6174675
time: 5.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.7145071, 3.7002277
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.4839706, 3.4695206
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.3600464, 3.3599715
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0048647, 5.0129633
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.2289162, 3.2112761
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -2.8947630, 2.9040179
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.8063793, 3.8068695
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.0645924, 4.0724134
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4557
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 5820

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4557

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5870854, upper bound: 1.5983933
time: 4.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5686277, upper bound: 1.6168508
time: 5.06 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.6987543, 3.7010870
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.4718323, 3.4696302
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.3364220, 3.3221660
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0030251, 5.0094547
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.2018881, 3.1952353
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -2.8870540, 2.9066076
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.7982931, 3.7961545
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.0194836, 4.0383072
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7224522, 3.7017117
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4557
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 884

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4557

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5860621, upper bound: 1.5990751
time: 5.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5676045, upper bound: 1.6175349
time: 4.83 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 24.58 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 24.58
Output dim: 9, lower bound: -1.6166572, upper bound: 1.5688439
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 24.58
Output dim: 9, lower bound: -1.6174283, upper bound: 1.5680481
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 24.58
Output dim: 9, lower bound: -1.6146287, upper bound: 1.5678499
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 24.58
Output dim: 9, lower bound: -1.6173582, upper bound: 1.5678481
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 24.58
Output dim: 9, lower bound: -1.5691333, upper bound: 1.6168884
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 24.58
Output dim: 9, lower bound: -1.5685537, upper bound: 1.6174675
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 24.58
Output dim: 9, lower bound: -1.5870854, upper bound: 1.5983933
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 24.58
Output dim: 9, lower bound: -1.5686277, upper bound: 1.6168508
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 24.58
Output dim: 9, lower bound: -1.5860621, upper bound: 1.5990751
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 24.58
Output dim: 9, lower bound: -1.5676045, upper bound: 1.6175349

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.6949863, 3.7126365
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.5041065, 3.5051270
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.2791605, 3.2909436
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0460720, 5.0180416
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.1814566, 3.2000647
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -2.9174252, 2.9026318
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.8253689, 3.8379383
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.0492153, 4.0186510
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.6896830, 3.7110319
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4610

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6174245, upper bound: 1.5674638
time: 5.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6168453, upper bound: 1.5680429
time: 5.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.7043304, 3.7102151
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.4718275, 3.4836302
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.2880716, 3.3129001
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0139160, 5.0043077
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.1627903, 3.1814623
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -2.9048615, 2.8874645
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.8086634, 3.7998323
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.0271597, 4.0047398
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.6892624, 3.7067337
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 4610

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4597

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.6165857, upper bound: 1.5678458
time: 5.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6173560, upper bound: 1.5670489
time: 5.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.6983910, 3.6800413
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.5109406, 3.4821911
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.2953091, 3.2705154
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0149794, 5.0095387
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.1643915, 3.1562934
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -2.8836875, 2.8700495
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.8130436, 3.8254375
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.0112724, 4.0332470
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7117233, 3.6965857
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 884

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5687227, upper bound: 1.6167792
time: 5.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5687246, upper bound: 1.6140493
time: 5.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.6915474, 3.6868792
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.5040302, 3.4891081
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.2940826, 3.2717414
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0203047, 5.0042219
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.1639032, 3.1567874
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -2.8771663, 2.8765678
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.8116512, 3.8268285
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.0110130, 4.0335083
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7087879, 3.6995211
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4608
type: RSZ, layer: 1, pos: 884

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4608

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5668752, upper bound: 1.6174618
time: 5.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5685478, upper bound: 1.6157907
time: 4.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.7143745, 3.6984844
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.4823542, 3.4673653
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.3337851, 3.3249741
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0031605, 5.0106878
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.2045441, 3.1787977
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -2.8836756, 2.8957033
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.8062820, 3.8154635
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.0518589, 4.0628586
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7254972
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.59 seconds
Binary search (step 2): status=Status.UNKNOWN, k_low=6, k_high=6, k_mid=6, eps_mid=0.0234375, abs_max=2.6592936515808105
rel_dist={9: [-1.6176245637311917, 1.6176249981850015]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01953125
execution time: 2414.25 seconds
