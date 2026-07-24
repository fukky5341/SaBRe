## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 1.579334386
Search space: {k/256 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-13.3209705, -8.9732313, -13.3209705, -8.9732313, -4.3477392, 4.3477392)
1: (-7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.7961807, 3.7961807)
2: (-10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807)
3: (-12.5703182, -9.4160767, -12.5703182, -9.4160767, -3.1542416, 3.1542416)
4: (5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.4022894, 3.4022894)
5: (-8.9787197, -5.6989894, -8.9787197, -5.6989894, -3.2797303, 3.2797303)
6: (-12.5030499, -8.9509478, -12.5030499, -8.9509478, -3.3759890, 3.3759892)
7: (-5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.9533715, 2.9533715)
8: (-1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666)
9: (-6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.7556219, 2.7556219)

## BASE Result
execution time: IAR + LP analysis = 14.82 + 34.42 = 49.23 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -2.5504220, upper bound: 2.5504211


# Binary Search by BASE starts (time budget: 3550.77 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=3.402289390563965
rel_dist={4: [-1.981378173882315, 1.9813780589154808]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=3.2979745864868164
rel_dist={4: [-1.6115652775740719, 1.611567303353243]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=3.17470645904541
rel_dist={4: [-1.3305644078822736, 1.3305637963562233]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=3.2363405227661133
rel_dist={4: [-1.4769317174548258, 1.4769330892355903]}

## Binary Search Result
Binary search time: 208.84 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Relational Split (RS_dual_Z) starts
Time budget: 3341.93 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6250
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 90

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 6250

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0912099, upper bound: 2.0912102
time: 5.95 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0912099, upper bound: 2.0912102
time: 5.89 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 12.05 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 12.05
Output dim: 4, lower bound: -2.0912099, upper bound: 2.0912102
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 12.05
Output dim: 4, lower bound: -2.0912099, upper bound: 2.0912102

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.9978418, 4.0290322
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.6917505, 3.6912489
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -3.0764132, 3.0565333
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.4022894, 3.4022894
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.9920821, 2.9899774
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.7588153, 2.7812335
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.9533715, 2.9533715
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.7556219, 2.7556219

Time for backsubstitution: 14.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 90

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 495

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0911795, upper bound: 2.0534514
time: 5.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0534512, upper bound: 2.0911798
time: 5.48 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -4.0290327, 3.9978409
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.6912489, 3.6917500
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -3.0565333, 3.0764132
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.4022894, 3.4022894
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.9899783, 2.9920824
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.7812333, 2.7588153
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.9533715, 2.9533715
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.7556219, 2.7556219

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 90

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 495

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0911795, upper bound: 2.0534515
time: 5.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0534512, upper bound: 2.0911798
time: 5.47 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 26.12 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 26.12
Output dim: 4, lower bound: -2.0911795, upper bound: 2.0534514
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 26.12
Output dim: 4, lower bound: -2.0534512, upper bound: 2.0911798
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 26.12
Output dim: 4, lower bound: -2.0911795, upper bound: 2.0534515
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 26.12
Output dim: 4, lower bound: -2.0534512, upper bound: 2.0911798

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -4.0070024, 4.0414419
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.6309490, 3.6622515
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -3.1133857, 3.0838282
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.4022894, 3.4022894
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.9829063, 2.9672060
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.7784910, 2.7957635
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.9478154, 2.9481051
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.7556219, 2.7556219

Time for backsubstitution: 14.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 90

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0911763, upper bound: 2.0532329
time: 5.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0404321, upper bound: 2.0534474
time: 5.20 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -4.0102506, 4.0381932
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.6627541, 3.6304464
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -3.1037083, 3.0935056
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.4022894, 3.4022894
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.9693108, 2.9808013
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.7733450, 2.8009095
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.9533715, 2.9382935
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.7556219, 2.7556219

Time for backsubstitution: 14.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 90

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0534469, upper bound: 2.0404326
time: 4.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0532326, upper bound: 2.0911776
time: 4.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -4.0381932, 4.0102501
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.6304474, 3.6627531
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -3.0935059, 3.1037080
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.4022894, 3.4022894
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.9808016, 2.9693110
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.8009095, 2.7733452
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.9382930, 2.9533715
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.7556219, 2.7556219

Time for backsubstitution: 14.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 90

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0911763, upper bound: 2.0532328
time: 4.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0404321, upper bound: 2.0534475
time: 4.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -4.0414433, 4.0070019
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.6622524, 3.6309476
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -3.0838284, 3.1133854
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.4022894, 3.4022894
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.9672060, 2.9829063
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.7957635, 2.7784913
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.9481053, 2.9478161
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.7556219, 2.7556219

Time for backsubstitution: 14.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 90

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0534469, upper bound: 2.0404322
time: 4.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0532326, upper bound: 2.0911767
time: 4.72 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 23.97 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.97
Output dim: 4, lower bound: -2.0911763, upper bound: 2.0532329
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.97
Output dim: 4, lower bound: -2.0404321, upper bound: 2.0534474
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.97
Output dim: 4, lower bound: -2.0534469, upper bound: 2.0404326
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.97
Output dim: 4, lower bound: -2.0532326, upper bound: 2.0911776
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.97
Output dim: 4, lower bound: -2.0911763, upper bound: 2.0532328
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.97
Output dim: 4, lower bound: -2.0404321, upper bound: 2.0534475
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.97
Output dim: 4, lower bound: -2.0534469, upper bound: 2.0404322
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.97
Output dim: 4, lower bound: -2.0532326, upper bound: 2.0911767

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -4.0077553, 4.0425253
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.6294184, 3.6601233
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -3.1199150, 3.0883811
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.4022894, 3.4022894
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.9834490, 2.9672003
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.7749386, 2.7932470
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.9504776, 2.9499688
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.7556219, 2.7556219

Time for backsubstitution: 14.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.38 seconds

### Candidate
type: RSZ, layer: 3, pos: 1690

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0640619, upper bound: 2.0258331
time: 4.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0639307, upper bound: 2.0259868
time: 4.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -4.0080853, 4.0421944
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.6288195, 3.6594968
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -3.1179380, 3.0903568
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.4022894, 3.4022894
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.9829006, 2.9677479
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.7739601, 2.7922106
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.9496803, 2.9491544
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.7556219, 2.7556219

Time for backsubstitution: 14.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.39 seconds

### Candidate
type: RSZ, layer: 3, pos: 1690

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0129500, upper bound: 2.0260511
time: 6.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0127883, upper bound: 2.0262032
time: 4.45 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -4.0110035, 4.0392766
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.6599989, 3.6283183
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -3.1102366, 3.0980585
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.4022894, 3.4022894
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.9698534, 2.9807956
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.7697926, 2.7963781
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.9533715, 2.9401572
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.7556219, 2.7556219

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.39 seconds

### Candidate
type: RSZ, layer: 3, pos: 1690

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0262032, upper bound: 2.0127882
time: 7.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0260514, upper bound: 2.0129501
time: 11.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -4.0113335, 4.0389462
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.6606245, 3.6289163
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -3.1082611, 3.1000352
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.4022894, 3.4022894
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.9693050, 2.9813433
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.7708282, 2.7973566
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.9533715, 2.9409540
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.7556219, 2.7556219

Time for backsubstitution: 14.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.40 seconds

### Candidate
type: RSZ, layer: 3, pos: 1690

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0259848, upper bound: 2.0639310
time: 5.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0258327, upper bound: 2.0640639
time: 4.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -4.0389462, 4.0113335
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.6289167, 3.6606250
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -3.1000352, 3.1082609
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.4022894, 3.4022894
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.9813433, 2.9693053
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.7973566, 2.7708287
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.9409542, 2.9533715
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.7556219, 2.7556219

Time for backsubstitution: 14.92 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.39 seconds

### Candidate
type: RSZ, layer: 3, pos: 1690

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0640619, upper bound: 2.0258331
time: 5.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0639307, upper bound: 2.0259868
time: 5.07 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -4.0392761, 4.0110030
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.6283178, 3.6599979
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -3.0980587, 3.1102366
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.4022894, 3.4022894
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.9807959, 2.9698529
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.7963781, 2.7697923
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.9401579, 2.9533715
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.7556219, 2.7556219

Time for backsubstitution: 14.69 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.40 seconds

### Candidate
type: RSZ, layer: 3, pos: 1690

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0129500, upper bound: 2.0260511
time: 8.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0127883, upper bound: 2.0262032
time: 5.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -4.0421944, 4.0080853
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.6594973, 3.6288199
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -3.0903568, 3.1179383
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.4022894, 3.4022894
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.9677477, 2.9829006
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.7922106, 2.7739599
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.9491549, 2.9496799
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.7556219, 2.7556219

Time for backsubstitution: 14.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.39 seconds

### Candidate
type: RSZ, layer: 3, pos: 1690

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0262032, upper bound: 2.0127882
time: 6.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0260514, upper bound: 2.0129502
time: 5.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -4.0425262, 4.0077543
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.6601229, 3.6294174
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -3.0883808, 3.1199150
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.4022894, 3.4022894
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.9672003, 2.9834485
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.7932472, 2.7749383
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.9499693, 2.9504769
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.7556219, 2.7556219

Time for backsubstitution: 14.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.39 seconds

### Candidate
type: RSZ, layer: 3, pos: 1690

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0259848, upper bound: 2.0639310
time: 4.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0258327, upper bound: 2.0640639
time: 4.31 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 23.89 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.89
Output dim: 4, lower bound: -2.0640619, upper bound: 2.0258331
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.89
Output dim: 4, lower bound: -2.0639307, upper bound: 2.0259868
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.89
Output dim: 4, lower bound: -2.0129500, upper bound: 2.0260511
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.89
Output dim: 4, lower bound: -2.0127883, upper bound: 2.0262032
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.89
Output dim: 4, lower bound: -2.0262032, upper bound: 2.0127882
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.89
Output dim: 4, lower bound: -2.0260514, upper bound: 2.0129501
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.89
Output dim: 4, lower bound: -2.0259848, upper bound: 2.0639310
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.89
Output dim: 4, lower bound: -2.0258327, upper bound: 2.0640639
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.89
Output dim: 4, lower bound: -2.0640619, upper bound: 2.0258331
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.89
Output dim: 4, lower bound: -2.0639307, upper bound: 2.0259868
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.89
Output dim: 4, lower bound: -2.0129500, upper bound: 2.0260511
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.89
Output dim: 4, lower bound: -2.0127883, upper bound: 2.0262032
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.89
Output dim: 4, lower bound: -2.0262032, upper bound: 2.0127882
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.89
Output dim: 4, lower bound: -2.0260514, upper bound: 2.0129502
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.89
Output dim: 4, lower bound: -2.0259848, upper bound: 2.0639310
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.89
Output dim: 4, lower bound: -2.0258327, upper bound: 2.0640639

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.9893017, 3.9334912
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.6407766, 3.5906386
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -3.1137590, 3.0566883
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.4022894, 3.4022894
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.9721146, 2.9428596
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.7414269, 2.7237420
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.9417057, 2.9192255
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.7556219, 2.7556219

Time for backsubstitution: 14.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0289819, upper bound: 1.9967415
time: 4.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0360811, upper bound: 1.9922880
time: 5.06 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.8987207, 4.0217566
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.5599337, 3.6696854
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -3.0882225, 3.0814271
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.4022894, 3.4022894
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.9591074, 2.9553480
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.7054334, 2.7584078
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.9197330, 2.9406786
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.7556219, 2.7556219

Time for backsubstitution: 14.92 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0285947, upper bound: 1.9968372
time: 5.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0359943, upper bound: 1.9925239
time: 5.50 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.9896307, 3.9331608
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.6401892, 3.5900121
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -3.1117935, 3.0586643
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.4022894, 3.4022894
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.9715643, 2.9434073
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.7404485, 2.7227056
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.9409161, 2.9184110
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.7556219, 2.7556219

Time for backsubstitution: 14.68 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.9810202, upper bound: 1.9969590
time: 6.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.9834533, upper bound: 1.9925060
time: 4.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.8990507, 4.0214276
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.5593348, 3.6690564
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -3.0862455, 3.0833921
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.4022894, 3.4022894
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.9585600, 2.9558959
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.7044549, 2.7573686
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.9189358, 2.9398613
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.7556219, 2.7556219

Time for backsubstitution: 14.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.9807829, upper bound: 1.9970552
time: 4.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.9833645, upper bound: 1.9927413
time: 5.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.9902363, 3.9302425
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.6695585, 3.5588336
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -3.1032724, 3.0663657
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.4022894, 3.4022894
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.9580011, 2.9564550
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.7349505, 2.7268732
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.9493837, 2.9094138
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.7556219, 2.7556219

Time for backsubstitution: 14.73 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.9927417, upper bound: 1.9833647
time: 6.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.9970547, upper bound: 1.9807826
time: 5.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.9019690, 4.0208220
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.5905142, 3.6396871
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -3.0785441, 3.0919139
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.4022894, 3.4022894
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.9455118, 2.9694598
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.7002873, 2.7628670
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.9279337, 2.9313931
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.7556219, 2.7556219

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.9925059, upper bound: 1.9834535
time: 5.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.9969588, upper bound: 1.9810200
time: 4.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.9905653, 3.9299116
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.6701880, 3.5594316
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -3.1013069, 3.0683424
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.4022894, 3.4022894
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.9574528, 2.9570026
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.7359896, 2.7278516
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.9502010, 2.9102106
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.7556219, 2.7556219

Time for backsubstitution: 14.66 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.9925238, upper bound: 2.0359947
time: 5.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.9968370, upper bound: 2.0285945
time: 7.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.9022999, 4.0204930
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.5911398, 3.6402745
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -3.0765681, 3.0938790
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.4022894, 3.4022894
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.9449644, 2.9700096
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.7013240, 2.7638450
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.9287481, 2.9321830
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.7556219, 2.7556219

Time for backsubstitution: 14.90 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.9922879, upper bound: 2.0360813
time: 4.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.9967412, upper bound: 2.0289823
time: 4.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -4.0204935, 3.9022999
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.6402750, 3.5911403
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -3.0938787, 3.0765681
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.4022894, 3.4022894
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.9700098, 2.9449646
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.7638450, 2.7013237
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.9321823, 2.9287481
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.7556219, 2.7556219

Time for backsubstitution: 14.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0289819, upper bound: 1.9967415
time: 4.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0360811, upper bound: 1.9922881
time: 4.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.9299116, 3.9905653
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.5594320, 3.6701870
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -3.0683422, 3.1013069
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.4022894, 3.4022894
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.9570026, 2.9574528
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.7278514, 2.7359896
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.9102106, 2.9502013
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.7556219, 2.7556219

Time for backsubstitution: 14.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0285947, upper bound: 1.9968371
time: 5.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -2.0359943, upper bound: 1.9925239
time: 4.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -4.0208225, 3.9019690
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.6396875, 3.5905137
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -3.0919142, 3.0785441
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.4022894, 3.4022894
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.9694595, 2.9455123
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.7628670, 2.7002873
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.9313927, 2.9279337
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.7556219, 2.7556219

Time for backsubstitution: 14.68 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.9810202, upper bound: 1.9969590
time: 6.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.9834533, upper bound: 1.9925060
time: 5.10 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.9302425, 3.9902363
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.5588331, 3.6695580
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -3.0663657, 3.1032722
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.4022894, 3.4022894
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.9564552, 2.9580009
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.7268729, 2.7349503
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.9094133, 2.9493840
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.7556219, 2.7556219

Time for backsubstitution: 14.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.9807829, upper bound: 1.9970553
time: 4.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.9833645, upper bound: 1.9927413
time: 8.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -4.0214281, 3.8990512
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.6690569, 3.5593352
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -3.0833921, 3.0862455
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.4022894, 3.4022894
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.9558954, 2.9585600
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.7573686, 2.7044549
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.9398613, 2.9189365
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.7556219, 2.7556219

Time for backsubstitution: 14.68 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.9927417, upper bound: 1.9833647
time: 6.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.9970547, upper bound: 1.9807826
time: 5.20 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.9331608, 3.9896307
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.5900126, 3.6401887
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -3.0586643, 3.1117938
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.4022894, 3.4022894
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.9434071, 2.9715648
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.7227054, 2.7404487
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.9184113, 2.9409161
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.7556219, 2.7556219

Time for backsubstitution: 14.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.9925059, upper bound: 1.9834534
time: 4.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.9969588, upper bound: 1.9810201
time: 4.47 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -4.0217562, 3.8987207
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.6696863, 3.5599332
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -3.0814271, 3.0882223
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.4022894, 3.4022894
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.9553480, 2.9591079
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.7584081, 2.7054334
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.9406786, 2.9197335
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.7556219, 2.7556219

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.20 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=3.402289390563965
rel_dist={4: [-2.091210306910777, 2.0912106090343423]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6250
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 90

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 6250

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7405443, upper bound: 1.7405442
time: 6.39 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7405443, upper bound: 1.7405442
time: 5.65 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 12.26 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 12.26
Output dim: 4, lower bound: -1.7405443, upper bound: 1.7405442
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 12.26
Output dim: 4, lower bound: -1.7405443, upper bound: 1.7405442

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.6406221, 3.6584449
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.4723406, 3.4720535
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.8437343, 2.8323743
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.3596191, 3.3595877
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.7681684, 2.7669654
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.4307857, 2.4435961
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.7821517, 2.7767096
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6402464, 2.6370716

Time for backsubstitution: 14.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 90

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 495

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7405267, upper bound: 1.7188964
time: 5.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7188963, upper bound: 1.7405288
time: 4.98 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.6584463, 3.6406212
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.4720545, 3.4723401
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.8323741, 2.8437343
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.3595877, 3.3596191
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.7669649, 2.7681684
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.4435964, 2.4307857
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.7767100, 2.7821512
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6370716, 2.6402464

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 90

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 495

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7405267, upper bound: 1.7188962
time: 5.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7188963, upper bound: 1.7405288
time: 5.25 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 25.00 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 25.00
Output dim: 4, lower bound: -1.7405267, upper bound: 1.7188964
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 25.00
Output dim: 4, lower bound: -1.7188963, upper bound: 1.7405288
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 25.00
Output dim: 4, lower bound: -1.7405267, upper bound: 1.7188962
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 25.00
Output dim: 4, lower bound: -1.7188963, upper bound: 1.7405288

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.6497827, 3.6694622
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.4115372, 3.4294257
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.8765593, 2.8596692
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.3552094, 3.3542991
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.7531652, 2.7441940
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.4482565, 2.4581261
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.7484488, 2.7486138
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6608095, 2.6618004

Time for backsubstitution: 14.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 90

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7405250, upper bound: 1.7187687
time: 4.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7110570, upper bound: 1.7188960
time: 4.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.6516385, 3.6676059
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.4297123, 3.4112511
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.8710294, 2.8651991
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.3543301, 3.3551788
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.7453966, 2.7519629
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.4453154, 2.4610667
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.7540555, 2.7430072
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6649752, 2.6576347

Time for backsubstitution: 14.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 90

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7188938, upper bound: 1.7110571
time: 6.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7187686, upper bound: 1.7405251
time: 6.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.6676068, 3.6516385
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.4112511, 3.4297118
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.8651991, 2.8710291
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.3551788, 3.3543301
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.7519627, 2.7453971
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.4610667, 2.4453156
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.7430072, 2.7540555
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6576352, 2.6649752

Time for backsubstitution: 14.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 90

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7405250, upper bound: 1.7187686
time: 4.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7110570, upper bound: 1.7188960
time: 4.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.6694627, 3.6497822
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.4294262, 3.4115376
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.8596697, 2.8765590
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.3542995, 3.3552098
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.7441940, 2.7531657
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.4581261, 2.4482563
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.7486138, 2.7484488
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6618004, 2.6608095

Time for backsubstitution: 14.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 90

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7188938, upper bound: 1.7110571
time: 5.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7187686, upper bound: 1.7405251
time: 7.71 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 27.92 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.92
Output dim: 4, lower bound: -1.7405250, upper bound: 1.7187687
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.92
Output dim: 4, lower bound: -1.7110570, upper bound: 1.7188960
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.92
Output dim: 4, lower bound: -1.7188938, upper bound: 1.7110571
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.92
Output dim: 4, lower bound: -1.7187686, upper bound: 1.7405251
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.92
Output dim: 4, lower bound: -1.7405250, upper bound: 1.7187686
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.92
Output dim: 4, lower bound: -1.7110570, upper bound: 1.7188960
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.92
Output dim: 4, lower bound: -1.7188938, upper bound: 1.7110571
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.92
Output dim: 4, lower bound: -1.7187686, upper bound: 1.7405251

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.6505356, 3.6704040
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.4097509, 3.4272971
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.8822412, 2.8642220
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.3561621, 3.3554759
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.7534733, 2.7441883
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.4447031, 2.4551654
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.7507687, 2.7504776
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6614752, 2.6626320

Time for backsubstitution: 14.69 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.39 seconds

### Candidate
type: RSZ, layer: 3, pos: 1690

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7178706, upper bound: 1.6960789
time: 6.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7179072, upper bound: 1.6960140
time: 5.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.6507244, 3.6702151
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.4094095, 3.4269395
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.8811121, 2.8653512
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.3559256, 3.3552518
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.7531595, 2.7445014
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.4441442, 2.4545732
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.7503128, 2.7500122
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6616383, 2.6624660

Time for backsubstitution: 14.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.38 seconds

### Candidate
type: RSZ, layer: 3, pos: 1690

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6882645, upper bound: 1.6962060
time: 4.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6883470, upper bound: 1.6961414
time: 5.19 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.6523914, 3.6685476
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.4272261, 3.4091229
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.8767109, 2.8697519
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.3552828, 3.3558946
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.7457037, 2.7519572
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.4417629, 2.4569547
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.7554541, 2.7448709
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6656408, 2.6584635

Time for backsubstitution: 14.54 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.39 seconds

### Candidate
type: RSZ, layer: 3, pos: 1690

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6961418, upper bound: 1.6883446
time: 5.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6962061, upper bound: 1.6882643
time: 4.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.6525803, 3.6683588
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.4275846, 3.4094644
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.8755817, 2.8708816
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.3555079, 3.3561311
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.7453909, 2.7522702
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.4423552, 2.4575138
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.7559195, 2.7453263
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6658063, 2.6583004

Time for backsubstitution: 14.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.39 seconds

### Candidate
type: RSZ, layer: 3, pos: 1690

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6960144, upper bound: 1.7179074
time: 5.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6960790, upper bound: 1.7178707
time: 4.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.6683578, 3.6525803
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.4094648, 3.4275842
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.8708816, 2.8755820
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.3561316, 3.3555074
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.7522707, 2.7453914
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.4575138, 2.4423549
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.7453270, 2.7559192
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6583004, 2.6658063

Time for backsubstitution: 14.67 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.39 seconds

### Candidate
type: RSZ, layer: 3, pos: 1690

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7178706, upper bound: 1.6960785
time: 6.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7179072, upper bound: 1.6960139
time: 5.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.6685486, 3.6523914
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.4091234, 3.4272256
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.8697519, 2.8767109
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.3558950, 3.3552823
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.7519569, 2.7457042
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.4569550, 2.4417627
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.7448711, 2.7554538
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6584635, 2.6656408

Time for backsubstitution: 14.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.39 seconds

### Candidate
type: RSZ, layer: 3, pos: 1690

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6882645, upper bound: 1.6962060
time: 4.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6883449, upper bound: 1.6961413
time: 5.23 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.6702156, 3.6507239
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.4269400, 3.4094095
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.8653512, 2.8811119
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.3552523, 3.3559260
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.7445011, 2.7531600
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.4545727, 2.4441440
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.7500124, 2.7503126
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6624660, 2.6616383

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.40 seconds

### Candidate
type: RSZ, layer: 3, pos: 1690

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6961418, upper bound: 1.6883444
time: 5.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6962061, upper bound: 1.6882643
time: 4.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.6704044, 3.6505346
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.4272966, 3.4097514
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.8642220, 2.8822415
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.3554764, 3.3561625
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.7441883, 2.7534730
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.4551649, 2.4447033
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.7504778, 2.7507679
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6626320, 2.6614752

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.39 seconds

### Candidate
type: RSZ, layer: 3, pos: 1690

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6960144, upper bound: 1.6962066
time: 9.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6960790, upper bound: 1.7178707
time: 4.76 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 29.09 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 29.09
Output dim: 4, lower bound: -1.7178706, upper bound: 1.6960789
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.09
Output dim: 4, lower bound: -1.7179072, upper bound: 1.6960140
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 29.09
Output dim: 4, lower bound: -1.6882645, upper bound: 1.6962060
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.09
Output dim: 4, lower bound: -1.6883470, upper bound: 1.6961414
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 29.09
Output dim: 4, lower bound: -1.6961418, upper bound: 1.6883446
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.09
Output dim: 4, lower bound: -1.6962061, upper bound: 1.6882643
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 29.09
Output dim: 4, lower bound: -1.6960144, upper bound: 1.7179074
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.09
Output dim: 4, lower bound: -1.6960790, upper bound: 1.7178707
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 29.09
Output dim: 4, lower bound: -1.7178706, upper bound: 1.6960785
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.09
Output dim: 4, lower bound: -1.7179072, upper bound: 1.6960139
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 29.09
Output dim: 4, lower bound: -1.6882645, upper bound: 1.6962060
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.09
Output dim: 4, lower bound: -1.6883449, upper bound: 1.6961413
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 29.09
Output dim: 4, lower bound: -1.6961418, upper bound: 1.6883444
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.09
Output dim: 4, lower bound: -1.6962061, upper bound: 1.6882643
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 29.09
Output dim: 4, lower bound: -1.6960144, upper bound: 1.6962066
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.09
Output dim: 4, lower bound: -1.6960790, upper bound: 1.7178707

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.5932617, 3.5613699
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.3864622, 3.3578129
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.8651409, 2.8325295
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.3111944, 3.3238821
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.7365646, 2.7198477
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.3957658, 2.3856604
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.7325802, 2.7197342
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6598825, 2.6614118

Time for backsubstitution: 14.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6961167, upper bound: 1.6776291
time: 5.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7000512, upper bound: 1.6751181
time: 4.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.5415010, 3.6118073
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.3402662, 3.4029822
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.8505487, 2.8466659
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.3240585, 3.3105087
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.7291327, 2.7269838
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.3751984, 2.4054694
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.7200241, 2.7319932
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6602554, 2.6610208

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6960017, upper bound: 1.6775359
time: 5.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7001052, upper bound: 1.6751062
time: 4.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.5934496, 3.5611811
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.3861265, 3.3574548
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.8640184, 2.8336585
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.3109579, 3.3236585
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.7362509, 2.7201607
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.3952069, 2.3850682
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.7321281, 2.7192688
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6600451, 2.6612463

Time for backsubstitution: 14.70 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6682113, upper bound: 1.6777585
time: 5.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6694094, upper bound: 1.6752464
time: 5.51 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.5416899, 3.6116190
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.3399248, 3.4026232
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.8494196, 2.8477888
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.3238239, 3.3102841
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.7288198, 2.7272971
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.3746390, 2.4048755
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.7195692, 2.7315261
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6604180, 2.6608553

Time for backsubstitution: 14.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6682510, upper bound: 1.6776658
time: 7.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6695194, upper bound: 1.6752367
time: 6.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.5937958, 3.5595136
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.4029093, 3.3396387
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.8591490, 2.8380594
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.3103151, 3.3237925
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.7284994, 2.7276165
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.3920650, 2.3874497
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.7369671, 2.7141275
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6640296, 2.6572437

Time for backsubstitution: 14.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6752347, upper bound: 1.6695190
time: 4.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6776656, upper bound: 1.6682508
time: 5.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.5433569, 3.6112733
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.3577414, 3.3858404
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.8450184, 2.8526583
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.3236895, 3.3109274
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.7213640, 2.7350478
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.3722577, 2.4080176
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.7247105, 2.7266874
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6644206, 2.6568708

Time for backsubstitution: 14.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6752460, upper bound: 1.6694092
time: 5.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6777584, upper bound: 1.6682113
time: 4.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.5939837, 3.5593247
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.4032698, 3.3399801
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.8580256, 2.8391888
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.3105392, 3.3240280
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.7281866, 2.7279296
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.3926592, 2.3880088
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.7374344, 2.7145829
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6641955, 2.6570807

Time for backsubstitution: 14.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6751068, upper bound: 1.7001049
time: 7.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6775359, upper bound: 1.6960015
time: 6.17 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.5435467, 3.6110854
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.3580999, 3.3861761
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.8438892, 2.8537812
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.3239126, 3.3111639
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.7210512, 2.7353621
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.3728499, 2.4085765
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.7251759, 2.7271385
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6645865, 2.6567078

Time for backsubstitution: 14.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6751182, upper bound: 1.7000511
time: 5.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6776289, upper bound: 1.6961170
time: 5.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.6110849, 3.5435462
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.3861761, 3.3580995
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.8537812, 2.8438892
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.3111639, 3.3239136
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.7353621, 2.7210507
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.4085765, 2.3728499
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.7271385, 2.7251759
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6567078, 2.6645865

Time for backsubstitution: 14.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6961167, upper bound: 1.6776288
time: 4.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7000512, upper bound: 1.6751181
time: 5.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.5593252, 3.5939837
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.3399801, 3.4032688
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.8391886, 2.8580258
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.3240271, 3.3105397
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.7279291, 2.7281868
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.3880086, 2.3926592
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.7145824, 2.7374349
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6570807, 2.6641951

Time for backsubstitution: 14.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6960017, upper bound: 1.6775360
time: 5.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.7001052, upper bound: 1.6751062
time: 4.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.6112738, 3.5433574
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.3858404, 3.3577414
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.8526583, 2.8450184
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.3109274, 3.3236895
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.7350473, 2.7213635
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.4080176, 2.3722577
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.7266865, 2.7247105
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6568704, 2.6644206

Time for backsubstitution: 14.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6682113, upper bound: 1.6777586
time: 4.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6694094, upper bound: 1.6752480
time: 4.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.5595140, 3.5937958
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.3396387, 3.4029098
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.8380594, 2.8591487
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.3237925, 3.3103151
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.7276163, 2.7284999
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.3874497, 2.3920650
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.7141275, 2.7369678
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6572437, 2.6640301

Time for backsubstitution: 14.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6682510, upper bound: 1.6776658
time: 6.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6695194, upper bound: 1.6752351
time: 5.49 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.6116190, 3.5416899
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.4026232, 3.3399253
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.8477888, 2.8494194
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.3102837, 3.3238239
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.7272968, 2.7288194
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.4048753, 2.3746390
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.7315254, 2.7195692
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6608553, 2.6604185

Time for backsubstitution: 14.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6752347, upper bound: 1.6695193
time: 5.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6776656, upper bound: 1.6682509
time: 4.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.5611811, 3.5934496
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.3574553, 3.3861270
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.8336582, 2.8640182
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.3236589, 3.3109584
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.7201605, 2.7362509
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.3850684, 2.3952072
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.7192688, 2.7321289
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6612458, 2.6600451

Time for backsubstitution: 14.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6752460, upper bound: 1.6694106
time: 4.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6777584, upper bound: 1.6682113
time: 5.13 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.6118078, 3.5415010
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.4029818, 3.3402667
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.8466659, 2.8505487
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.3105087, 3.3240590
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.7269840, 2.7291324
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.4054694, 2.3751984
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.7319927, 2.7200246
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6610208, 2.6602554

Time for backsubstitution: 14.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6751068, upper bound: 1.6777580
time: 10.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6775359, upper bound: 1.6960021
time: 4.80 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 29.52 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 29.52
Output dim: 4, lower bound: -1.6961167, upper bound: 1.6776291
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 29.52
Output dim: 4, lower bound: -1.7000512, upper bound: 1.6751181
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 29.52
Output dim: 4, lower bound: -1.6960017, upper bound: 1.6775359
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 29.52
Output dim: 4, lower bound: -1.7001052, upper bound: 1.6751062
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 29.52
Output dim: 4, lower bound: -1.6682113, upper bound: 1.6777585
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 29.52
Output dim: 4, lower bound: -1.6694094, upper bound: 1.6752464
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 29.52
Output dim: 4, lower bound: -1.6682510, upper bound: 1.6776658
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 29.52
Output dim: 4, lower bound: -1.6695194, upper bound: 1.6752367
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 29.52
Output dim: 4, lower bound: -1.6752347, upper bound: 1.6695190
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 29.52
Output dim: 4, lower bound: -1.6776656, upper bound: 1.6682508
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 29.52
Output dim: 4, lower bound: -1.6752460, upper bound: 1.6694092
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 29.52
Output dim: 4, lower bound: -1.6777584, upper bound: 1.6682113
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 29.52
Output dim: 4, lower bound: -1.6751068, upper bound: 1.7001049
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 29.52
Output dim: 4, lower bound: -1.6775359, upper bound: 1.6960015
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 29.52
Output dim: 4, lower bound: -1.6751182, upper bound: 1.7000511
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 29.52
Output dim: 4, lower bound: -1.6776289, upper bound: 1.6961170
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 29.52
Output dim: 4, lower bound: -1.6961167, upper bound: 1.6776288
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 29.52
Output dim: 4, lower bound: -1.7000512, upper bound: 1.6751181
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 29.52
Output dim: 4, lower bound: -1.6960017, upper bound: 1.6775360
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 29.52
Output dim: 4, lower bound: -1.7001052, upper bound: 1.6751062
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 29.52
Output dim: 4, lower bound: -1.6682113, upper bound: 1.6777586
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 29.52
Output dim: 4, lower bound: -1.6694094, upper bound: 1.6752480
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 29.52
Output dim: 4, lower bound: -1.6682510, upper bound: 1.6776658
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 29.52
Output dim: 4, lower bound: -1.6695194, upper bound: 1.6752351
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 29.52
Output dim: 4, lower bound: -1.6752347, upper bound: 1.6695193
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 29.52
Output dim: 4, lower bound: -1.6776656, upper bound: 1.6682509
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 29.52
Output dim: 4, lower bound: -1.6752460, upper bound: 1.6694106
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 29.52
Output dim: 4, lower bound: -1.6777584, upper bound: 1.6682113
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 29.52
Output dim: 4, lower bound: -1.6751068, upper bound: 1.6777580
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 29.52
Output dim: 4, lower bound: -1.6775359, upper bound: 1.6960021
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.52
Output dim: 4, lower bound: -1.6960790, upper bound: 1.7178707
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=3.359607696533203
rel_dist={4: [-1.74054476319957, 1.7405458869370563]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6250
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 90

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 6250

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6115648, upper bound: 1.6115668
time: 5.17 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6115648, upper bound: 1.6115668
time: 4.77 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 10.14 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 10.14
Output dim: 4, lower bound: -1.6115648, upper bound: 1.6115668
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 10.14
Output dim: 4, lower bound: -1.6115648, upper bound: 1.6115668

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.5215483, 3.5349159
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.3992033, 3.3989882
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7887115, 2.7949197
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.7661748, 2.7576547
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.2979774, 3.2979541
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.6935301, 2.6926281
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.3214426, 2.3310504
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.7156959, 2.7116146
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.5823317, 2.5799503

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 90

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 495

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6115511, upper bound: 1.5954346
time: 4.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5954347, upper bound: 1.6115534
time: 5.17 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.5349169, 3.5215480
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.3989878, 3.3992038
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7949200, 2.7887113
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.7576547, 2.7661746
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.2979546, 3.2979774
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.6926279, 2.6935303
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.3310504, 2.3214426
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.7116141, 2.7156954
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.5799503, 2.5823317

Time for backsubstitution: 14.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 90

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 495

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5954369, upper bound: 1.5954343
time: 6.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5954347, upper bound: 1.6115534
time: 5.20 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 26.69 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 26.69
Output dim: 4, lower bound: -1.6115511, upper bound: 1.5954346
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 26.69
Output dim: 4, lower bound: -1.5954347, upper bound: 1.6115534
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 26.69
Output dim: 4, lower bound: -1.5954369, upper bound: 1.5954343
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 26.69
Output dim: 4, lower bound: -1.5954347, upper bound: 1.6115534

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.5307088, 3.5454693
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.3384018, 3.3518167
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.7976170, 2.7849495
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.2933483, 3.2926650
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.6765852, 2.6698568
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.3381777, 2.3455803
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.6819930, 2.6821165
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6028948, 2.6036377

Time for backsubstitution: 14.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 90

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6115499, upper bound: 1.5953406
time: 4.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5895300, upper bound: 1.5954348
time: 4.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.5321012, 3.5440769
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.3520317, 3.3381858
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.7934694, 2.7890971
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.2926884, 3.2933249
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.6707592, 2.6756835
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.3359728, 2.3477857
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.6861978, 2.6779118
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2211523, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6060185, 2.6005139

Time for backsubstitution: 14.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 90

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5954326, upper bound: 1.5895323
time: 5.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5953408, upper bound: 1.6115502
time: 4.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.5440774, 3.5321016
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.3381863, 3.3520317
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.7890973, 2.7934694
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.2933254, 3.2926888
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.6756830, 2.6707590
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.3477859, 2.3359725
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.6779113, 2.6861982
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2211518
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6005139, 2.6060190

Time for backsubstitution: 14.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 90

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6115499, upper bound: 1.5953406
time: 4.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5895300, upper bound: 1.5954348
time: 4.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.5454698, 3.5307093
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.3518181, 3.3384013
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.7849498, 2.7976170
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.2926655, 3.2933483
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.6698570, 2.6765857
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.3455801, 2.3381779
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.6821160, 2.6819930
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6036377, 2.6028948

Time for backsubstitution: 14.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 90

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5954326, upper bound: 1.5895323
time: 5.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5953408, upper bound: 1.6115502
time: 4.82 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 24.69 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.69
Output dim: 4, lower bound: -1.6115499, upper bound: 1.5953406
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.69
Output dim: 4, lower bound: -1.5895300, upper bound: 1.5954348
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.69
Output dim: 4, lower bound: -1.5954326, upper bound: 1.5895323
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.69
Output dim: 4, lower bound: -1.5953408, upper bound: 1.6115502
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.69
Output dim: 4, lower bound: -1.6115499, upper bound: 1.5953406
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.69
Output dim: 4, lower bound: -1.5895300, upper bound: 1.5954348
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.69
Output dim: 4, lower bound: -1.5954326, upper bound: 1.5895323
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.69
Output dim: 4, lower bound: -1.5953408, upper bound: 1.6115502

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.5314617, 3.5463638
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.3365297, 3.3496890
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.8030167, 2.7895024
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.2943010, 3.2937860
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.6768141, 2.6698511
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.3346252, 2.3424716
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.6841993, 2.6839805
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6035604, 2.6044278

Time for backsubstitution: 14.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.39 seconds

### Candidate
type: RSZ, layer: 3, pos: 1690

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5908304, upper bound: 1.5745843
time: 4.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5908715, upper bound: 1.5745261
time: 4.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.5316029, 3.5462217
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.3362722, 3.3494201
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.8021698, 2.7903492
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.2941236, 3.2936177
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.6765795, 2.6700859
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.3342056, 2.3420274
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.6838570, 2.6836314
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6036825, 2.6043034

Time for backsubstitution: 14.69 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.39 seconds

### Candidate
type: RSZ, layer: 3, pos: 1690

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5686511, upper bound: 1.5746830
time: 5.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5687194, upper bound: 1.5746249
time: 6.24 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.5328541, 3.5449715
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.3496351, 3.3360581
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.7988691, 2.7936499
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.2936411, 3.2941003
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.6709881, 2.6756778
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.3324194, 2.3438134
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.6877127, 2.6797752
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2205467, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6066847, 2.6013017

Time for backsubstitution: 14.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.39 seconds

### Candidate
type: RSZ, layer: 3, pos: 1690

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5746229, upper bound: 1.5687190
time: 5.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5746810, upper bound: 1.5686486
time: 5.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.5329952, 3.5448294
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.3499041, 3.3363142
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.7980223, 2.7944970
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.2938099, 3.2942777
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.6707535, 2.6759124
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.3328638, 2.3442328
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.6880617, 2.6801171
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2204618, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6068087, 2.6011796

Time for backsubstitution: 14.84 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.40 seconds

### Candidate
type: RSZ, layer: 3, pos: 1690

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5745262, upper bound: 1.5908712
time: 4.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5745846, upper bound: 1.5908299
time: 5.20 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.5448303, 3.5329957
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.3363142, 3.3499036
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.7944970, 2.7980223
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.2942781, 3.2938094
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.6759119, 2.6707532
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.3442326, 2.3328638
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.6801176, 2.6880617
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2204618
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6011796, 2.6068087

Time for backsubstitution: 14.70 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.40 seconds

### Candidate
type: RSZ, layer: 3, pos: 1690

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5908304, upper bound: 1.5745843
time: 4.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5908715, upper bound: 1.5745260
time: 6.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.5449715, 3.5328541
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.3360586, 3.3496351
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.7936497, 2.7988691
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.2941008, 3.2936411
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.6756773, 2.6709881
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.3438129, 2.3324196
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.6797762, 2.6877127
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2205467
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6013017, 2.6066847

Time for backsubstitution: 14.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.41 seconds

### Candidate
type: RSZ, layer: 3, pos: 1690

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5686511, upper bound: 1.5746830
time: 5.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5687194, upper bound: 1.5746249
time: 5.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.5462227, 3.5316033
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.3494196, 3.3362727
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.7903490, 2.8021698
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.2936182, 3.2941232
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.6700859, 2.6765800
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.3420277, 2.3342056
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.6836319, 2.6838570
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6043034, 2.6036825

Time for backsubstitution: 14.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.40 seconds

### Candidate
type: RSZ, layer: 3, pos: 1690

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5746229, upper bound: 1.5687190
time: 5.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5746810, upper bound: 1.5686486
time: 4.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.5463638, 3.5314617
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.3496885, 3.3365293
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7975807, 2.7975807
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.7895021, 2.8030169
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.2937860, 3.2943006
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.6698513, 2.6768146
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.3424711, 2.3346250
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.6839809, 2.6841984
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2218666, 3.2218666
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6044278, 2.6035604

Time for backsubstitution: 14.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.40 seconds

### Candidate
type: RSZ, layer: 3, pos: 1690

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5745262, upper bound: 1.5908712
time: 4.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5745846, upper bound: 1.5908299
time: 5.14 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 24.86 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.86
Output dim: 4, lower bound: -1.5908304, upper bound: 1.5745843
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.86
Output dim: 4, lower bound: -1.5908715, upper bound: 1.5745261
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 24.86
Output dim: 4, lower bound: -1.5686511, upper bound: 1.5746830
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 24.86
Output dim: 4, lower bound: -1.5687194, upper bound: 1.5746249
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 24.86
Output dim: 4, lower bound: -1.5746229, upper bound: 1.5687190
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 24.86
Output dim: 4, lower bound: -1.5746810, upper bound: 1.5686486
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.86
Output dim: 4, lower bound: -1.5745262, upper bound: 1.5908712
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.86
Output dim: 4, lower bound: -1.5745846, upper bound: 1.5908299
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.86
Output dim: 4, lower bound: -1.5908304, upper bound: 1.5745843
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.86
Output dim: 4, lower bound: -1.5908715, upper bound: 1.5745260
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 24.86
Output dim: 4, lower bound: -1.5686511, upper bound: 1.5746830
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 24.86
Output dim: 4, lower bound: -1.5687194, upper bound: 1.5746249
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 24.86
Output dim: 4, lower bound: -1.5746229, upper bound: 1.5687190
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 24.86
Output dim: 4, lower bound: -1.5746810, upper bound: 1.5686486
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.86
Output dim: 4, lower bound: -1.5745262, upper bound: 1.5908712
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.86
Output dim: 4, lower bound: -1.5745846, upper bound: 1.5908299

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.4612484, 3.4373295
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.3016920, 3.2802043
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7753959, 2.7690644
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.7822685, 2.7578096
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.2493334, 3.2588491
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.6580486, 2.6455104
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.2805457, 2.2729666
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.6628714, 2.6532373
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.1984234, 3.2002592
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6020608, 2.6032076

Time for backsubstitution: 14.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5736005, upper bound: 1.5598004
time: 4.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5765162, upper bound: 1.5579789
time: 6.50 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.4224281, 3.4751575
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.2670450, 3.3140812
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7659569, 2.7780812
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.7713242, 2.7684121
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.2589808, 3.2488189
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.6524744, 2.6508627
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.2651200, 2.2878234
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.6534548, 2.6624317
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2005920, 3.1978431
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6023402, 2.6029143

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5735239, upper bound: 1.5597212
time: 5.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5765724, upper bound: 1.5579660
time: 9.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.4617901, 3.4357955
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.3142958, 3.2668300
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7718730, 2.7721655
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.7769318, 2.7628043
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.2488413, 3.2589579
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.6517649, 2.6515718
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.2782154, 2.2747278
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.6665125, 2.6493740
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.1935101, 3.2049246
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6052957, 2.5999594

Time for backsubstitution: 14.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5579666, upper bound: 1.5765719
time: 5.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5597213, upper bound: 1.5735235
time: 5.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.4239616, 3.4746161
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.2804193, 3.3014770
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7628560, 2.7816043
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.7663298, 2.7737486
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.2588720, 3.2493100
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.6464128, 2.6571462
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.2633586, 2.2901535
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.6573181, 2.6587906
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.1959267, 3.2027559
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6055889, 2.5996799

Time for backsubstitution: 14.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5527418, upper bound: 1.5598215
time: 6.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5598007, upper bound: 1.5736000
time: 5.05 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.4746161, 3.4239619
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.3014765, 3.2804193
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7816043, 2.7628560
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.7737484, 2.7663298
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.2493105, 3.2588720
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.6571465, 2.6464126
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.2901535, 2.2633588
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.6587906, 2.6573181
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2027559, 3.1959267
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.5996795, 2.6055889

Time for backsubstitution: 14.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5736005, upper bound: 1.5598004
time: 5.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5765162, upper bound: 1.5579792
time: 6.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.4357958, 3.4617898
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.2668295, 3.3142962
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7721653, 2.7718730
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.7628040, 2.7769320
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.2589579, 3.2488422
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.6515722, 2.6517649
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.2747278, 2.2782154
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.6493740, 2.6665125
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2049246, 3.1935105
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.5999594, 2.6052952

Time for backsubstitution: 14.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5735239, upper bound: 1.5597210
time: 4.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5765724, upper bound: 1.5579663
time: 10.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.4751577, 3.4224277
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.3140821, 3.2670445
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7780809, 2.7659571
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.7684121, 2.7713242
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.2488184, 3.2589817
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.6508627, 2.6524739
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.2878232, 2.2651200
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.6624317, 2.6534548
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.1978436, 3.2005920
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6029143, 2.6023407

Time for backsubstitution: 14.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5579666, upper bound: 1.5765718
time: 4.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5527421, upper bound: 1.5580764
time: 5.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.3209705, -8.9732313, -13.3209705, -8.9732313, -3.4373293, 3.4612484
1: -7.3048649, -3.5086842, -7.3048649, -3.5086842, -3.2802038, 3.3016915
2: -10.0570240, -7.2594433, -10.0570240, -7.2594433, -2.7690644, 2.7753959
3: -12.5703182, -9.4160767, -12.5703182, -9.4160767, -2.7578096, 2.7822685
4: 5.3104191, 8.7127085, 5.3104191, 8.7127085, -3.2588491, 3.2493334
5: -8.9787197, -5.6989894, -8.9787197, -5.6989894, -2.6455107, 2.6580484
6: -12.5030499, -8.9509478, -12.5030499, -8.9509478, -2.2729669, 2.2805457
7: -5.7039032, -2.7505317, -5.7039032, -2.7505317, -2.6532364, 2.6628714
8: -1.2158751, 2.0059915, -1.2158751, 2.0059915, -3.2002592, 3.1984234
9: -6.5885048, -3.8328829, -6.5885048, -3.8328829, -2.6032081, 2.6020608

Time for backsubstitution: 14.69 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1109
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 2132
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1145
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2642
type: RSZ, layer: 3, pos: 1746
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2860
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2334
type: RSZ, layer: 3, pos: 1676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1241
type: RSZ, layer: 3, pos: 2564
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 709
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1852
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2384
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 2118
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1396
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2371
type: RSZ, layer: 3, pos: 2570
type: RSZ, layer: 3, pos: 409
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1782
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 310
type: RSZ, layer: 3, pos: 1103
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 2314
type: RSZ, layer: 3, pos: 611
type: RSZ, layer: 3, pos: 2594
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2572
type: RSZ, layer: 3, pos: 2369
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 221
type: RSZ, layer: 3, pos: 1165
type: RSZ, layer: 3, pos: 1199
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5579795, upper bound: 1.5765156
time: 5.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5598007, upper bound: 1.5735998
time: 5.23 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 25.55 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 25.55
Output dim: 4, lower bound: -1.5736005, upper bound: 1.5598004
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 25.55
Output dim: 4, lower bound: -1.5765162, upper bound: 1.5579789
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 25.55
Output dim: 4, lower bound: -1.5735239, upper bound: 1.5597212
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 25.55
Output dim: 4, lower bound: -1.5765724, upper bound: 1.5579660
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 25.55
Output dim: 4, lower bound: -1.5579666, upper bound: 1.5765719
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 25.55
Output dim: 4, lower bound: -1.5597213, upper bound: 1.5735235
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 25.55
Output dim: 4, lower bound: -1.5527418, upper bound: 1.5598215
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 25.55
Output dim: 4, lower bound: -1.5598007, upper bound: 1.5736000
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 25.55
Output dim: 4, lower bound: -1.5736005, upper bound: 1.5598004
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 25.55
Output dim: 4, lower bound: -1.5765162, upper bound: 1.5579792
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 25.55
Output dim: 4, lower bound: -1.5735239, upper bound: 1.5597210
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 25.55
Output dim: 4, lower bound: -1.5765724, upper bound: 1.5579663
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 25.55
Output dim: 4, lower bound: -1.5579666, upper bound: 1.5765718
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 25.55
Output dim: 4, lower bound: -1.5527421, upper bound: 1.5580764
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 25.55
Output dim: 4, lower bound: -1.5579795, upper bound: 1.5765156
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 25.55
Output dim: 4, lower bound: -1.5598007, upper bound: 1.5735998
Binary search (step 2): status=Status.VERIFIED, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=3.2979745864868164
rel_dist={4: [-1.6115652775740719, 1.611567303353243]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01171875
execution time: 2244.17 seconds
