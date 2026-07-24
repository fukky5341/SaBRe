## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 1.1665576092
Search space: {k/256 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-9.0259066, -5.5622492, -9.0259066, -5.5622492, -3.4636574, 3.4636574)
1: (-6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.6174438, 2.6174438)
2: (8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.6077518, 2.6077518)
3: (-6.1232843, -2.8826065, -6.1232843, -2.8826065, -3.2406778, 3.2406778)
4: (-11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.8509436, 3.8509436)
5: (-13.6636562, -10.1825514, -13.6636562, -10.1825514, -3.4811049, 3.4811049)
6: (-15.6556635, -12.3171921, -15.6556635, -12.3171921, -3.2803464, 3.2803464)
7: (-5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.5209539, 3.5209539)
8: (-1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.3452835, 2.3452835)
9: (-7.3109250, -4.0054374, -7.3109250, -4.0054374, -3.3054876, 3.3054876)

## BASE Result
execution time: IAR + LP analysis = 14.78 + 33.51 = 48.29 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3551.71 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=2.5009913444519043
rel_dist={2: [-1.4846913114101667, 1.4846931118005955]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=2.2642369270324707
rel_dist={2: [-1.168894797061638, 1.1688945587998152]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=2.106400489807129
rel_dist={2: [-0.9214908145334242, 0.9214909870646206]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=2.1853187084198
rel_dist={2: [-1.048994002949069, 1.0489933180675663]}

## Binary Search Result
Binary search time: 221.18 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Relational Split (RS_random_Z) starts
Time budget: 3330.53 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 5843

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4616

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5710314, upper bound: 1.5739989
time: 8.06 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5739987, upper bound: 1.5710331
time: 8.82 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 16.89 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 16.89
Output dim: 2, lower bound: -1.5710314, upper bound: 1.5739989
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 16.89
Output dim: 2, lower bound: -1.5739987, upper bound: 1.5710331

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -3.1037884, 3.1092148
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.4937468, 2.4988472
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.5795999, 2.5802269
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -3.2406778, 3.2406778
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.4196348, 3.4169846
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.9679861, 2.9654984
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.7503595, 2.7476425
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.4741755, 3.4746242
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.2999616, 2.3000240
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -3.0080500, 3.0106649

Time for backsubstitution: 15.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 131

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 929

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5703165, upper bound: 1.5714710
time: 9.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5685041, upper bound: 1.5732836
time: 9.25 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -3.1064348, 3.1037879
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.4962492, 2.4937468
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.5799088, 2.5795996
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -3.2406778, 3.2406778
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.4169855, 3.4182835
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.9654980, 2.9667301
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.7476416, 2.7489347
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.4743948, 3.4741759
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.2999711, 2.2999616
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -3.0093145, 3.0080500

Time for backsubstitution: 14.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6231

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5739940, upper bound: 1.5685697
time: 11.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5715370, upper bound: 1.5710268
time: 11.19 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 37.43 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 37.43
Output dim: 2, lower bound: -1.5703165, upper bound: 1.5714710
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 37.43
Output dim: 2, lower bound: -1.5685041, upper bound: 1.5732836
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 37.43
Output dim: 2, lower bound: -1.5739940, upper bound: 1.5685697
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 37.43
Output dim: 2, lower bound: -1.5715370, upper bound: 1.5710268

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -3.0990849, 3.1073513
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.4890890, 2.4970078
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.5795951, 2.5811715
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -3.2406778, 3.2406778
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.4169087, 3.4101009
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.9674969, 2.9642687
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.7493696, 2.7451515
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.4715881, 3.4680848
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.2993259, 2.2997713
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -3.0099726, 3.0106549

Time for backsubstitution: 14.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 520

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4666

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5659356, upper bound: 1.5714596
time: 9.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5703052, upper bound: 1.5670927
time: 16.20 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -3.1019249, 3.1045113
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.4919081, 2.4941900
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.5805440, 2.5802226
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -3.2406778, 3.2406778
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.4127507, 3.4142590
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.9667568, 2.9650097
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.7478685, 2.7466521
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.4676361, 3.4720364
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.2997088, 2.2993884
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -3.0080395, 3.0125875

Time for backsubstitution: 14.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 520

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4639

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5684462, upper bound: 1.5588177
time: 10.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5540829, upper bound: 1.5732267
time: 5.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -3.0945425, 3.0953627
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.4969921, 2.4942510
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.5760269, 2.5741227
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -3.2406778, 3.2406778
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.4143043, 3.4131408
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.9541874, 2.9504662
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.7486434, 2.7504106
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.4748344, 3.4756293
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.2998486, 2.2973790
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -3.0075054, 3.0049253

Time for backsubstitution: 14.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 498

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 131

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5739883, upper bound: 1.5665706
time: 7.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5719951, upper bound: 1.5685638
time: 9.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -3.0980091, 3.0918970
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.4967527, 2.4944904
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.5744333, 2.5757165
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -3.2406778, 3.2406778
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.4118409, 3.4156046
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.9492321, 2.9554214
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.7491183, 2.7499361
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.4758492, 3.4746146
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.2973895, 2.2998381
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -3.0061903, 3.0062408

Time for backsubstitution: 14.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 131

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5691599, upper bound: 1.5685996
time: 7.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5691106, upper bound: 1.5686488
time: 6.46 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 28.89 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 28.89
Output dim: 2, lower bound: -1.5659356, upper bound: 1.5714596
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 28.89
Output dim: 2, lower bound: -1.5703052, upper bound: 1.5670927
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 28.89
Output dim: 2, lower bound: -1.5684462, upper bound: 1.5588177
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 28.89
Output dim: 2, lower bound: -1.5540829, upper bound: 1.5732267
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 28.89
Output dim: 2, lower bound: -1.5739883, upper bound: 1.5665706
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 28.89
Output dim: 2, lower bound: -1.5719951, upper bound: 1.5685638
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 28.89
Output dim: 2, lower bound: -1.5691599, upper bound: 1.5685996
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 28.89
Output dim: 2, lower bound: -1.5691106, upper bound: 1.5686488

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -3.0980453, 3.1080399
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.4679737, 2.4650793
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.5653043, 2.5728323
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -3.2406778, 3.2406778
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.4141517, 3.4062095
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.9672556, 2.9640961
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.7333055, 2.7348213
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.4532714, 3.4384756
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.2954793, 2.2927337
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -3.0029039, 3.0056334

Time for backsubstitution: 14.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4654

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 115

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5659225, upper bound: 1.5698432
time: 8.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5643191, upper bound: 1.5714470
time: 19.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -3.0997734, 3.1063128
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.4571609, 2.4758914
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.5712557, 2.5668807
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -3.2406778, 3.2406778
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.4130187, 3.4073429
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.9673243, 2.9640265
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.7390389, 2.7290874
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.4419780, 3.4497681
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.2922878, 2.2959247
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -3.0049505, 3.0035863

Time for backsubstitution: 14.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6111

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 520

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5703017, upper bound: 1.5649421
time: 11.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5681562, upper bound: 1.5670872
time: 8.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -3.1177197, 3.1315036
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.4925056, 2.4891469
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.5715642, 2.5675740
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -3.2406778, 3.2406778
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.4068012, 3.4100409
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.9807701, 2.9747014
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.7461247, 2.7582793
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.4636555, 3.4650202
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.3058825, 2.3107076
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.9913006, 3.0018516

Time for backsubstitution: 14.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 131

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5684394, upper bound: 1.5568138
time: 5.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5664560, upper bound: 1.5588079
time: 9.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -3.1289167, 3.1203060
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.4868646, 2.4947875
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.5678959, 2.5712428
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -3.2406778, 3.2406778
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.4085331, 3.4083090
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.9764490, 2.9790225
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.7594972, 2.7449069
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.4606199, 3.4680552
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.3110275, 2.3055620
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.9973040, 2.9958482

Time for backsubstitution: 14.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 131

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5517084, upper bound: 1.5708007
time: 9.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5516590, upper bound: 1.5708500
time: 6.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -3.0945415, 3.0953622
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.4969921, 2.4942505
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.5760250, 2.5741200
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -3.2406778, 3.2406778
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.4143043, 3.4131403
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.9541874, 2.9504652
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.7486429, 2.7504106
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.4748349, 3.4756284
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.2998481, 2.2973785
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -3.0075040, 3.0049229

Time for backsubstitution: 15.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 520

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5739848, upper bound: 1.5644216
time: 28.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5718393, upper bound: 1.5665676
time: 9.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -3.0945425, 3.0953612
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.4969921, 2.4942505
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.5760241, 2.5741212
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -3.2406778, 3.2406778
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.4143043, 3.4131403
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.9541874, 2.9504657
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.7486429, 2.7504101
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.4748330, 3.4756293
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.2998486, 2.2973785
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -3.0075030, 3.0049238

Time for backsubstitution: 14.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 115

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6111

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5719936, upper bound: 1.5657168
time: 8.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5691483, upper bound: 1.5685626
time: 7.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -3.0981092, 3.0918865
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.4967499, 2.4945226
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.5744333, 2.5757203
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -3.2406778, 3.2406778
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.4120083, 3.4155874
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.9492655, 2.9554214
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.7491255, 2.7499361
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.4758415, 3.4747000
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.2973971, 2.2998371
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -3.0063705, 3.0062222

Time for backsubstitution: 14.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 4666

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6191

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5691595, upper bound: 1.5675802
time: 5.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5681403, upper bound: 1.5685989
time: 23.48 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -3.0979977, 3.0918970
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.4967527, 2.4944868
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.5744333, 2.5757170
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -3.2406778, 3.2406778
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.4118233, 3.4156046
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.9492331, 2.9554214
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.7491179, 2.7499361
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.4758492, 3.4746056
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.2973886, 2.2998381
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -3.0061722, 3.0062408

Time for backsubstitution: 14.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 5843

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 520

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5691071, upper bound: 1.5664994
time: 7.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5669616, upper bound: 1.5686447
time: 12.01 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 34.60 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 34.60
Output dim: 2, lower bound: -1.5659225, upper bound: 1.5698432
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 34.60
Output dim: 2, lower bound: -1.5643191, upper bound: 1.5714470
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 34.60
Output dim: 2, lower bound: -1.5703017, upper bound: 1.5649421
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 34.60
Output dim: 2, lower bound: -1.5681562, upper bound: 1.5670872
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 34.60
Output dim: 2, lower bound: -1.5684394, upper bound: 1.5568138
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 34.60
Output dim: 2, lower bound: -1.5664560, upper bound: 1.5588079
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 34.60
Output dim: 2, lower bound: -1.5517084, upper bound: 1.5708007
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 34.60
Output dim: 2, lower bound: -1.5516590, upper bound: 1.5708500
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 34.60
Output dim: 2, lower bound: -1.5739848, upper bound: 1.5644216
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 34.60
Output dim: 2, lower bound: -1.5718393, upper bound: 1.5665676
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 34.60
Output dim: 2, lower bound: -1.5719936, upper bound: 1.5657168
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 34.60
Output dim: 2, lower bound: -1.5691483, upper bound: 1.5685626
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 34.60
Output dim: 2, lower bound: -1.5691595, upper bound: 1.5675802
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 34.60
Output dim: 2, lower bound: -1.5681403, upper bound: 1.5685989
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 34.60
Output dim: 2, lower bound: -1.5691071, upper bound: 1.5664994
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 34.60
Output dim: 2, lower bound: -1.5669616, upper bound: 1.5686447

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -3.0979719, 3.1079874
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.4680948, 2.4651642
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.5650706, 2.5724998
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -3.2406778, 3.2406778
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.4140749, 3.4061031
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.9670591, 2.9640627
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.7331581, 2.7347176
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.4539003, 3.4388680
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.2957926, 2.2929449
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -3.0024595, 3.0050173

Time for backsubstitution: 14.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 4656

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 520

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5659190, upper bound: 1.5692536
time: 7.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5637857, upper bound: 1.5676820
time: 11.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -3.0979939, 3.1079650
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.4680586, 2.4652014
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.5649719, 2.5725985
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -3.2406778, 3.2406778
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.4140444, 3.4061341
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.9672222, 2.9638996
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.7332020, 2.7346733
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.4536629, 3.4391055
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.2956905, 2.2930470
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -3.0022879, 3.0051885

Time for backsubstitution: 14.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 5843

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6231

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5643145, upper bound: 1.5689855
time: 9.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5618575, upper bound: 1.5714440
time: 8.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -3.0974360, 3.1050849
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.4557686, 2.4771819
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.5666814, 2.5604229
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -3.2406778, 3.2406778
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.4075341, 3.3996015
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.9697952, 2.9677596
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.7372160, 2.7281311
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.4427223, 3.4515738
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.2910681, 2.2954311
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -3.0044098, 3.0028219

Time for backsubstitution: 14.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 6111

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4639

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5702451, upper bound: 1.5505236
time: 8.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5558335, upper bound: 1.5648842
time: 5.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -3.0985441, 3.1039767
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.4584522, 2.4744997
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.5647984, 2.5623062
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -3.2406778, 3.2406778
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.4052777, 3.4018598
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.9710560, 2.9664979
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.7380819, 2.7272637
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.4437847, 3.4505115
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.2917943, 2.2947044
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -3.0041866, 3.0030451

Time for backsubstitution: 14.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 536

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5681496, upper bound: 1.5632487
time: 23.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5642986, upper bound: 1.5670807
time: 8.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -3.1177177, 3.1315031
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.4925056, 2.4891472
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.5715628, 2.5675721
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -3.2406778, 3.2406778
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.4068022, 3.4100409
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.9807701, 2.9747009
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.7461243, 2.7582793
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.4636555, 3.4650187
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.3058820, 2.3107076
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.9913006, 3.0018506

Time for backsubstitution: 14.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 4654

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5684319, upper bound: 1.5529173
time: 8.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5645420, upper bound: 1.5568059
time: 16.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -3.1177187, 3.1315017
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.4925065, 2.4891472
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.5715618, 2.5675731
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -3.2406778, 3.2406778
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.4068012, 3.4100409
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.9807692, 2.9747014
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.7461243, 2.7582803
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.4636536, 3.4650197
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.3058825, 2.3107071
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.9913006, 3.0018516

Time for backsubstitution: 14.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 536

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5640780, upper bound: 1.5563848
time: 6.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5640286, upper bound: 1.5564337
time: 8.15 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -3.1290178, 3.1202955
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.4868608, 2.4948199
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.5678959, 2.5712466
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -3.2406778, 3.2406778
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.4087019, 3.4082918
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.9764814, 2.9790225
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.7595043, 2.7449069
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.4606113, 3.4681420
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.3110352, 2.3055606
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.9974852, 2.9958310

Time for backsubstitution: 14.82 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=2.579909086227417
rel_dist={2: [-1.574020518019939, 1.5740203796460204]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 4666

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6170

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2849340, upper bound: 1.2819889
time: 18.27 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2819893, upper bound: 1.2849343
time: 8.75 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 27.04 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 27.04
Output dim: 2, lower bound: -1.2849340, upper bound: 1.2819889
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 27.04
Output dim: 2, lower bound: -1.2819893, upper bound: 1.2849343

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.8218918, 2.8256841
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2718697, 2.2739217
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.3332467, 2.3304453
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9856243, 2.9915018
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.0877771, 3.0887299
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.5948315, 2.5986223
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.3983388, 2.4080806
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.2936282, 3.2959590
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.1074028, 2.1036677
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7761230, 2.7737174

Time for backsubstitution: 14.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 4654

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 131

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2849335, upper bound: 1.2809328
time: 8.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2809351, upper bound: 1.2819884
time: 6.36 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.8256836, 2.8218923
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2739210, 2.2718697
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.3304453, 2.3332467
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9915018, 2.9856243
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.0887289, 3.0877781
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.5986223, 2.5948319
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.4080796, 2.3983393
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.2959590, 3.2936287
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.1036677, 2.1074023
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7737179, 2.7761226

Time for backsubstitution: 14.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4654

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6219

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2819853, upper bound: 1.2827150
time: 10.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2797798, upper bound: 1.2849299
time: 12.07 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 36.78 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 36.78
Output dim: 2, lower bound: -1.2849335, upper bound: 1.2809328
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 36.78
Output dim: 2, lower bound: -1.2809351, upper bound: 1.2819884
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 36.78
Output dim: 2, lower bound: -1.2819853, upper bound: 1.2827150
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 36.78
Output dim: 2, lower bound: -1.2797798, upper bound: 1.2849299

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.8218899, 2.8256831
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2718697, 2.2739220
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.3332453, 2.3304427
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9856200, 2.9914980
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.0877790, 3.0887299
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.5948315, 2.5986214
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.3983383, 2.4080806
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.2936277, 3.2959571
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.1074023, 2.1036677
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7761207, 2.7737155

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6231

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 929

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2844497, upper bound: 1.2796447
time: 7.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2836631, upper bound: 1.2804489
time: 31.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.8218918, 2.8256822
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2718697, 2.2739220
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.3332443, 2.3304434
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9856210, 2.9914970
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.0877790, 3.0887303
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.5948305, 2.5986218
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.3983383, 2.4080806
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.2936258, 3.2959576
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.1074023, 2.1036677
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7761207, 2.7737160

Time for backsubstitution: 14.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 4639

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 536

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2780389, upper bound: 1.2799469
time: 8.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2828121, upper bound: 1.2819859
time: 11.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.8085804, 2.8076401
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2780342, 2.2772548
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.3209877, 2.3199520
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9946003, 2.9896860
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.0772820, 3.0782342
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.6029749, 2.6005349
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.3801999, 2.3751063
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.2994757, 3.2982340
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.0789714, 2.0868297
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7750545, 2.7771435

Time for backsubstitution: 14.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 4656

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 536

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2797776, upper bound: 1.2806627
time: 8.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2799414, upper bound: 1.2827118
time: 5.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.8114319, 2.8047891
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2793064, 2.2759824
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.3171506, 2.3237891
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9955635, 2.9887238
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.0791855, 3.0763307
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.6043253, 2.5991845
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.3848481, 2.3704591
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.3005648, 3.2971454
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.0830946, 2.0827060
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7747378, 2.7774591

Time for backsubstitution: 14.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 6231

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4666

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2772744, upper bound: 1.2849258
time: 13.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2797745, upper bound: 1.2824272
time: 9.08 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 37.07 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 37.07
Output dim: 2, lower bound: -1.2844497, upper bound: 1.2796447
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 37.07
Output dim: 2, lower bound: -1.2836631, upper bound: 1.2804489
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 37.07
Output dim: 2, lower bound: -1.2780389, upper bound: 1.2799469
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 37.07
Output dim: 2, lower bound: -1.2828121, upper bound: 1.2819859
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 37.07
Output dim: 2, lower bound: -1.2797776, upper bound: 1.2806627
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 37.07
Output dim: 2, lower bound: -1.2799414, upper bound: 1.2827118
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 37.07
Output dim: 2, lower bound: -1.2772744, upper bound: 1.2849258
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 37.07
Output dim: 2, lower bound: -1.2797745, upper bound: 1.2824272

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.8171864, 2.8226018
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2672119, 2.2708745
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.3332410, 2.3309808
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9830985, 2.9898462
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.0832696, 3.0818453
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.5940256, 2.5973921
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.3967056, 2.4055896
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.2893453, 3.2894182
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.1067662, 2.1032505
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7772160, 2.7737060

Time for backsubstitution: 14.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 498

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5843

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2844487, upper bound: 1.2790739
time: 10.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2830968, upper bound: 1.2790732
time: 14.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.8188095, 2.8209791
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2688217, 2.2692642
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.3337831, 2.3304384
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9839673, 2.9889765
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.0808940, 3.0842214
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.5936022, 2.5978155
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.3958492, 2.4064469
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.2870879, 3.2916760
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.1069851, 2.1030316
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7761116, 2.7748103

Time for backsubstitution: 14.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 4639

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 520

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2836609, upper bound: 1.2792113
time: 11.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2824256, upper bound: 1.2804485
time: 11.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.8088360, 2.8175712
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2702894, 2.2713838
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.3296895, 2.3247187
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9820585, 2.9857616
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.0874891, 3.0885458
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.5920801, 2.5941901
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.3826370, 2.3983240
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.2906828, 3.2912159
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.0992208, 2.0985832
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7723842, 2.7713933

Time for backsubstitution: 14.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 929

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4656

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2645268, upper bound: 1.2694238
time: 10.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2674907, upper bound: 1.2799410
time: 11.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.8137798, 2.8126273
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2693319, 2.2723415
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.3275199, 2.3268886
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9798861, 2.9879379
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.0875940, 3.0884409
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.5903988, 2.5958710
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.3885822, 2.3923788
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.2888842, 3.2930145
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.1023183, 2.0954857
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7737994, 2.7699795

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 498

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5843

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2828110, upper bound: 1.2807742
time: 7.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2816074, upper bound: 1.2819848
time: 7.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.7955256, 2.7995300
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2764530, 2.2747169
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.3174329, 2.3142271
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9910402, 2.9839516
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.0769939, 3.0780520
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.6002254, 2.5961041
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.3644986, 2.3653507
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.2965322, 3.2934923
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.0707898, 2.0817471
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7713161, 2.7748199

Time for backsubstitution: 14.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 115

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2819742, upper bound: 1.2798276
time: 6.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2811558, upper bound: 1.2806556
time: 11.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.8004694, 2.7945852
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2754955, 2.2756743
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.3152628, 2.3163970
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9888659, 2.9861245
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.0770988, 3.0779467
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.5985441, 2.5977855
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.3704438, 2.3594041
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.2947335, 3.2952900
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.0738873, 2.0786486
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7727313, 2.7734056

Time for backsubstitution: 15.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 131

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6191

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2799412, upper bound: 1.2822041
time: 10.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2794194, upper bound: 1.2827115
time: 8.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.8103971, 2.8047404
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2535563, 2.2440526
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.3028593, 2.3128986
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9856062, 2.9804235
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.0759411, 3.0724382
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.6040821, 2.5989823
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.3687916, 2.3576808
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.2774029, 3.2675309
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.0778732, 2.0756626
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7676682, 2.7715597

Time for backsubstitution: 15.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 4639

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4656

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2772707, upper bound: 1.2744006
time: 12.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2667510, upper bound: 1.2849196
time: 5.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.8113832, 2.8037534
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2473774, 2.2502313
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.3062596, 2.3094978
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9872627, 2.9787669
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.0752945, 3.0730858
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.6041222, 2.5989423
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.3720703, 2.3544040
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.2709503, 3.2739835
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.0760512, 2.0774860
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7688384, 2.7703900

Time for backsubstitution: 14.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4639

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 536

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2797694, upper bound: 1.2803840
time: 6.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2777215, upper bound: 1.2824227
time: 6.63 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 28.03 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.03
Output dim: 2, lower bound: -1.2844487, upper bound: 1.2790739
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.03
Output dim: 2, lower bound: -1.2830968, upper bound: 1.2790732
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.03
Output dim: 2, lower bound: -1.2836609, upper bound: 1.2792113
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.03
Output dim: 2, lower bound: -1.2824256, upper bound: 1.2804485
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.03
Output dim: 2, lower bound: -1.2645268, upper bound: 1.2694238
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.03
Output dim: 2, lower bound: -1.2674907, upper bound: 1.2799410
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.03
Output dim: 2, lower bound: -1.2828110, upper bound: 1.2807742
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.03
Output dim: 2, lower bound: -1.2816074, upper bound: 1.2819848
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.03
Output dim: 2, lower bound: -1.2819742, upper bound: 1.2798276
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.03
Output dim: 2, lower bound: -1.2811558, upper bound: 1.2806556
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.03
Output dim: 2, lower bound: -1.2799412, upper bound: 1.2822041
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.03
Output dim: 2, lower bound: -1.2794194, upper bound: 1.2827115
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.03
Output dim: 2, lower bound: -1.2772707, upper bound: 1.2744006
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.03
Output dim: 2, lower bound: -1.2667510, upper bound: 1.2849196
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.03
Output dim: 2, lower bound: -1.2797694, upper bound: 1.2803840
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.03
Output dim: 2, lower bound: -1.2777215, upper bound: 1.2824227

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.7884016, 2.7986150
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2310972, 2.2407813
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.3375907, 2.3367684
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9914322, 2.9965901
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.0624747, 3.0540962
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.5835381, 2.5813322
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.3748856, 2.3794103
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.2726774, 3.2682314
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.0779772, 2.0792623
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7827301, 2.7778401

Time for backsubstitution: 15.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 520

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4616

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2815991, upper bound: 1.2790494
time: 6.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2844277, upper bound: 1.2762171
time: 6.07 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.7932005, 2.7938170
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2371178, 2.2347598
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.3390284, 2.3353310
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9898415, 2.9964428
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.0555196, 3.0610504
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.5779648, 2.5869055
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.3705273, 2.3837700
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.2681589, 3.2727499
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.0827780, 2.0744615
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7813492, 2.7792206

Time for backsubstitution: 15.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 498

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4666

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2805995, upper bound: 1.2790676
time: 7.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2830903, upper bound: 1.2765734
time: 27.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.8164749, 2.8192778
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2674294, 2.2694037
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.3284001, 2.3239794
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9845285, 2.9890718
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.0744433, 3.0764813
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.5960732, 2.6010079
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.3940244, 2.4051189
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.2878308, 3.2930264
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.1057653, 2.1022272
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7754755, 2.7740469

Time for backsubstitution: 15.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 6191

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6231

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2836582, upper bound: 1.2778127
time: 12.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2822603, upper bound: 1.2792092
time: 13.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.8171082, 2.8186445
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2689619, 2.2678709
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.3273239, 2.3250556
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9840631, 2.9895377
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.0731540, 3.0777717
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.5967941, 2.6002865
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.3945203, 2.4046235
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.2884383, 3.2924194
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.1061807, 2.1018119
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7753477, 2.7741742

Time for backsubstitution: 14.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4639

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2824216, upper bound: 1.2782185
time: 8.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2801829, upper bound: 1.2804431
time: 15.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.7939129, 2.8051276
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2706218, 2.2713156
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.3151054, 2.3072248
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9657040, 2.9721217
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.0683808, 3.0656347
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.5826330, 2.5863113
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.3692932, 2.3871937
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.2938957, 3.2922325
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.0790300, 2.0817528
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7775087, 2.7750778

Time for backsubstitution: 14.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 6231

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 520

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2780329, upper bound: 1.2681864
time: 10.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2767978, upper bound: 1.2694221
time: 9.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.7963924, 2.8026481
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2702203, 2.2717164
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.3121958, 2.3101344
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9684191, 2.9694057
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.0645776, 3.0694380
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.5842018, 2.5847430
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.3715076, 2.3849797
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.2916985, 3.2944283
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.0823903, 2.0783925
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7760687, 2.7765183

Time for backsubstitution: 14.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 498

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4666

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2649569, upper bound: 1.2799349
time: 8.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2674821, upper bound: 1.2774392
time: 8.24 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.7849951, 2.7886410
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2332172, 2.2422485
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.3318691, 2.3326755
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9882207, 2.9946818
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.0599351, 3.0538278
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.5786877, 2.5785861
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.3667622, 2.3661971
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.2662616, 3.2658725
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.0735292, 2.0714974
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7793117, 2.7741122

Time for backsubstitution: 15.06 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=2.3431549072265625
rel_dist={2: [-1.2849368558531555, 1.2849364906580067]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 6219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4654

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1688921, upper bound: 1.1682862
time: 5.72 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1682859, upper bound: 1.1688922
time: 7.07 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 12.81 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 12.81
Output dim: 2, lower bound: -1.1688921, upper bound: 1.1682862
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 12.81
Output dim: 2, lower bound: -1.1682859, upper bound: 1.1688922

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.7369213, 2.7360291
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2128778, 2.2126234
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2626734, 2.2620256
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9077044, 2.9126983
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -2.9842644, 2.9845600
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.4963026, 2.4942775
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.3218026, 2.3223882
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.2482548, 3.2466888
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.0677733, 2.0678835
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7127790, 2.7132587

Time for backsubstitution: 15.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 498

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 536

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1688894, upper bound: 1.1667621
time: 8.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1673681, upper bound: 1.1682839
time: 7.48 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.7360287, 2.7369204
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2126231, 2.2128778
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2620258, 2.2626734
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9126987, 2.9077048
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -2.9845591, 2.9842644
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.4942770, 2.4963026
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.3223882, 2.3218017
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.2466879, 3.2482553
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.0678835, 2.0677738
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7132587, 2.7127795

Time for backsubstitution: 14.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 520

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4639

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1682574, upper bound: 1.1629138
time: 5.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1623086, upper bound: 1.1688643
time: 25.44 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 45.78 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 45.78
Output dim: 2, lower bound: -1.1688894, upper bound: 1.1667621
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 45.78
Output dim: 2, lower bound: -1.1673681, upper bound: 1.1682839
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 45.78
Output dim: 2, lower bound: -1.1682574, upper bound: 1.1629138
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 45.78
Output dim: 2, lower bound: -1.1623086, upper bound: 1.1688643

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.7238655, 2.7266822
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2110567, 2.2100849
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2585759, 2.2563007
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9035997, 2.9069638
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -2.9839745, 2.9843493
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.4931316, 2.4898462
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.3061004, 2.3111458
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.2448616, 3.2419462
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.0595927, 2.0620260
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7090425, 2.7105827

Time for backsubstitution: 14.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 6170

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1670711, upper bound: 1.1648887
time: 5.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1670163, upper bound: 1.1649435
time: 6.23 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.7275734, 2.7229743
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2103386, 2.2108033
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2569485, 2.2579281
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9019699, 2.9085937
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -2.9840536, 2.9842706
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.4918709, 2.4911070
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.3105588, 2.3066864
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.2435131, 3.2432952
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.0619159, 2.0597029
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7101030, 2.7095218

Time for backsubstitution: 15.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6191

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1673679, upper bound: 1.1678591
time: 16.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1669469, upper bound: 1.1682832
time: 9.13 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.7518196, 2.7575078
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2099981, 2.2078359
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2509503, 2.2500257
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9303193, 2.9224997
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -2.9786100, 2.9790573
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.5058212, 2.5059943
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.3206415, 2.3257861
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.2409744, 3.2412405
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.0740576, 2.0761528
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.6965222, 2.6986156

Time for backsubstitution: 15.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 5843

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -1.1664391, upper bound: 1.1610412
time: 11.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -1.1663843, upper bound: 1.1610962
time: 11.21 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.7566185, 2.7527108
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2075815, 2.2102542
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2493777, 2.2515979
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9274936, 2.9253254
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -2.9793520, 2.9783154
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.5039692, 2.5078464
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.3263721, 2.3200550
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.2396736, 3.2425413
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.0762630, 2.0739479
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.6990952, 2.6960425

Time for backsubstitution: 14.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 131

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 929

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1617900, upper bound: 1.1677303
time: 5.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1612072, upper bound: 1.1683220
time: 6.08 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 26.86 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.86
Output dim: 2, lower bound: -1.1670711, upper bound: 1.1648887
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 26.86
Output dim: 2, lower bound: -1.1670163, upper bound: 1.1649435
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.86
Output dim: 2, lower bound: -1.1673679, upper bound: 1.1678591
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 26.86
Output dim: 2, lower bound: -1.1669469, upper bound: 1.1682832
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 26.86
Output dim: 2, lower bound: -1.1664391, upper bound: 1.1610412
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 26.86
Output dim: 2, lower bound: -1.1663843, upper bound: 1.1610962
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.86
Output dim: 2, lower bound: -1.1617900, upper bound: 1.1677303
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 26.86
Output dim: 2, lower bound: -1.1612072, upper bound: 1.1683220

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.7239037, 2.7266712
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2110538, 2.2100973
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2585754, 2.2563021
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9036322, 2.9069548
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -2.9840384, 2.9843330
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.4931459, 2.4898467
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.3061042, 2.3111458
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.2448530, 3.2419786
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.0595956, 2.0620251
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7091084, 2.7105637

Time for backsubstitution: 14.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 520

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6191

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1670710, upper bound: 1.1644683
time: 7.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1666475, upper bound: 1.1648886
time: 6.18 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.7238550, 2.7266822
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2110567, 2.2100818
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2585759, 2.2563007
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9035912, 2.9069638
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -2.9839592, 2.9843493
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.4931326, 2.4898462
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.3061004, 2.3111458
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.2448616, 3.2419381
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.0595918, 2.0620260
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7090235, 2.7105827

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 115

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4656

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -1.1584502, upper bound: 1.1569836
time: 11.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -1.1590542, upper bound: 1.1649396
time: 8.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.7189398, 2.7203021
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.1914539, 2.1942832
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2611790, 2.2578609
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.8321581, 2.8480825
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -2.9723310, 2.9708738
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.4932251, 2.4927745
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.3087478, 2.3051009
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.2372985, 3.2325258
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.0575700, 2.0524874
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7159266, 2.7167048

Time for backsubstitution: 14.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 115

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1673672, upper bound: 1.1665403
time: 7.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1650243, upper bound: 1.1678608
time: 10.24 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.7249022, 2.7143407
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.1938190, 2.1919181
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2568812, 2.2621589
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.8414583, 2.8387818
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -2.9706573, 2.9725475
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.4935379, 2.4924607
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.3089728, 2.3048744
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.2327437, 3.2370806
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.0546999, 2.0553570
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7172856, 2.7153449

Time for backsubstitution: 14.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 6170

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4656

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1669433, upper bound: 1.1603211
time: 5.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1589805, upper bound: 1.1682786
time: 8.11 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.7519150, 2.7492237
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2029238, 2.2068048
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2493734, 2.2519999
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9249721, 2.9234562
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -2.9742508, 2.9714317
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.5030565, 2.5066166
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.3245258, 2.3175659
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.2348247, 3.2359996
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.0756273, 2.0734763
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.6999130, 2.6960330

Time for backsubstitution: 14.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 520

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 536

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -1.1617877, upper bound: 1.1662225
time: 10.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1602777, upper bound: 1.1677275
time: 5.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.7531319, 2.7480078
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2041321, 2.2055972
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2497797, 2.2515931
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9256244, 2.9228039
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -2.9724693, 2.9732141
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.5027399, 2.5069337
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.3238831, 2.3182087
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.2331309, 3.2376928
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.0757914, 2.0733123
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.6990852, 2.6968613

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4666

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1593565, upper bound: 1.1683170
time: 6.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -1.1612018, upper bound: 1.1664830
time: 6.08 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 26.52 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.52
Output dim: 2, lower bound: -1.1670710, upper bound: 1.1644683
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.52
Output dim: 2, lower bound: -1.1666475, upper bound: 1.1648886
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 26.52
Output dim: 2, lower bound: -1.1584502, upper bound: 1.1569836
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 26.52
Output dim: 2, lower bound: -1.1590542, upper bound: 1.1649396
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.52
Output dim: 2, lower bound: -1.1673672, upper bound: 1.1665403
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.52
Output dim: 2, lower bound: -1.1650243, upper bound: 1.1678608
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.52
Output dim: 2, lower bound: -1.1669433, upper bound: 1.1603211
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.52
Output dim: 2, lower bound: -1.1589805, upper bound: 1.1682786
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 26.52
Output dim: 2, lower bound: -1.1617877, upper bound: 1.1662225
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.52
Output dim: 2, lower bound: -1.1602777, upper bound: 1.1677275
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.52
Output dim: 2, lower bound: -1.1593565, upper bound: 1.1683170
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 26.52
Output dim: 2, lower bound: -1.1612018, upper bound: 1.1664830

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.7152710, 2.7240005
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.1921692, 2.1935773
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2628064, 2.2562349
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.8338203, 2.8464441
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -2.9723148, 2.9709353
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.4944992, 2.4915133
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.3042917, 2.3095603
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.2386384, 3.2312088
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.0552502, 2.0548096
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7149320, 2.7177472

Time for backsubstitution: 14.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6111

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4666

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -1.1651956, upper bound: 1.1644629
time: 7.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1670657, upper bound: 1.1625931
time: 6.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.7212324, 2.7180390
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.1945343, 2.1912119
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2585087, 2.2605329
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.8431215, 2.8371434
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -2.9706402, 2.9726090
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.4948130, 2.4911995
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.3045187, 2.3093333
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.2340837, 3.2357645
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.0523806, 2.0576797
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7162919, 2.7163873

Time for backsubstitution: 15.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 4656

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6170

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1666454, upper bound: 1.1625763
time: 9.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -1.1643347, upper bound: 1.1648887
time: 11.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.7156849, 2.7165813
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.1876106, 2.1909242
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2603493, 2.2537858
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.8132792, 2.8265018
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -2.9671092, 2.9677644
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.4916162, 2.4918923
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.2949505, 2.2889028
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.2246623, 3.2214718
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.0577378, 2.0539913
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7159271, 2.7169886

Time for backsubstitution: 15.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4616

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -1.1644696, upper bound: 1.1646651
time: 6.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -1.1644696, upper bound: 1.1647201
time: 7.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.7152195, 2.7170472
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.1880951, 2.1904395
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2571039, 2.2570310
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.8105774, 2.8292036
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -2.9692206, 2.9656529
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.4923429, 2.4911661
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.2925491, 2.2913051
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.2262444, 3.2198896
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.0590739, 2.0526552
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7162113, 2.7167048

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4639

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6170

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -1.1627113, upper bound: 1.1655461
time: 14.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1637460, upper bound: 1.1678568
time: 5.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.7099791, 2.7012773
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.1940517, 2.1918502
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2415695, 2.2446647
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.8251019, 2.8244619
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -2.9505973, 2.9496355
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.4840918, 2.4841895
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.2956285, 2.2931890
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.2354069, 3.2380972
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.0345092, 2.0376863
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7220526, 2.7190309

Time for backsubstitution: 15.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 115

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1669371, upper bound: 1.1597154
time: 6.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -1.1663408, upper bound: 1.1603146
time: 6.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.7118378, 2.6994181
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.1937513, 2.1921506
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2393870, 2.2468472
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.8271379, 2.8224254
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -2.9477458, 2.9524879
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.4852686, 2.4830132
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.2972879, 2.2915287
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.2337599, 3.2397442
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.0370297, 2.0351658
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7209721, 2.7201109

Time for backsubstitution: 14.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 6111

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 498

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -1.1562377, upper bound: 1.1661412
time: 6.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1562377, upper bound: 1.1682740
time: 6.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.7425680, 2.7361689
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2003865, 2.2049839
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2436476, 2.2479019
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9192371, 2.9193501
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -2.9740410, 2.9711447
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.4986248, 2.5034461
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.3132820, 2.3018618
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.2300830, 3.2326059
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.0697703, 2.0652962
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.6972361, 2.6922951

Time for backsubstitution: 14.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 131

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4666

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1578355, upper bound: 1.1677231
time: 6.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -1.1602724, upper bound: 1.1659105
time: 5.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.7520914, 2.7477074
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.1768389, 2.1736708
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2354898, 2.2398539
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9156685, 2.9140902
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -2.9690638, 2.9693232
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.5024967, 2.5067205
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.3078194, 2.3046021
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.2083607, 3.2080836
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.0701203, 2.0662737
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.6920156, 2.6906695

Time for backsubstitution: 14.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4616

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5843

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1589563, upper bound: 1.1673059
time: 6.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1589574, upper bound: 1.1683187
time: 7.47 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 28.89 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 28.89
Output dim: 2, lower bound: -1.1651956, upper bound: 1.1644629
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 28.89
Output dim: 2, lower bound: -1.1670657, upper bound: 1.1625931
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 28.89
Output dim: 2, lower bound: -1.1666454, upper bound: 1.1625763
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 28.89
Output dim: 2, lower bound: -1.1643347, upper bound: 1.1648887
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 28.89
Output dim: 2, lower bound: -1.1644696, upper bound: 1.1646651
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 28.89
Output dim: 2, lower bound: -1.1644696, upper bound: 1.1647201
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 28.89
Output dim: 2, lower bound: -1.1627113, upper bound: 1.1655461
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 28.89
Output dim: 2, lower bound: -1.1637460, upper bound: 1.1678568
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 28.89
Output dim: 2, lower bound: -1.1669371, upper bound: 1.1597154
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 28.89
Output dim: 2, lower bound: -1.1663408, upper bound: 1.1603146
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 28.89
Output dim: 2, lower bound: -1.1562377, upper bound: 1.1661412
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 28.89
Output dim: 2, lower bound: -1.1562377, upper bound: 1.1682740
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 28.89
Output dim: 2, lower bound: -1.1578355, upper bound: 1.1677231
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 28.89
Output dim: 2, lower bound: -1.1602724, upper bound: 1.1659105
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 28.89
Output dim: 2, lower bound: -1.1589563, upper bound: 1.1673059
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 28.89
Output dim: 2, lower bound: -1.1589574, upper bound: 1.1683187

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.7149677, 2.7229571
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.1602426, 2.1662838
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2510667, 2.2419446
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.8251076, 2.8364882
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -2.9684229, 2.9675293
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.4942875, 2.4912715
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.2906814, 2.2934937
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.2090268, 3.2064357
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.0482125, 2.0491405
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7087407, 2.7106781

Time for backsubstitution: 14.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 6219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6231

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1670636, upper bound: 1.1615436
time: 6.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -1.1660161, upper bound: 1.1625911
time: 7.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.7121916, 2.7118421
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.1822200, 2.1804368
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2478986, 2.2478218
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.8213263, 2.8197560
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -2.9649029, 2.9675851
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.4720764, 2.4713063
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.2727709, 2.2848926
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.2200885, 3.2235174
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.0327449, 2.0352430
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7036796, 2.7019715

Time for backsubstitution: 15.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 4616

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1666425, upper bound: 1.1609417
time: 8.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -1.1620507, upper bound: 1.1625731
time: 8.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.7090235, 2.7080078
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.1773205, 2.1781256
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2443933, 2.2464216
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.7931914, 2.8074093
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -2.9641995, 2.9599171
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.4724493, 2.4684291
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.2681074, 2.2595582
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.2139964, 3.2058940
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.0366373, 2.0330191
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7017965, 2.7040944

Time for backsubstitution: 14.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 929

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 115

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1637398, upper bound: 1.1672542
time: 8.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1631403, upper bound: 1.1678502
time: 7.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.7099047, 2.7012124
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.1941509, 2.1919332
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2412806, 2.2443337
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.8249636, 2.8243032
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -2.9505043, 2.9495292
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.4838963, 2.4840641
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.2954798, 2.2930598
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.2359009, 3.2384896
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.0347643, 2.0378976
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7215099, 2.7184153

Time for backsubstitution: 14.95 seconds
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=2.2642369270324707
rel_dist={2: [-1.168894797061638, 1.1688945587998152]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2420.64 seconds
