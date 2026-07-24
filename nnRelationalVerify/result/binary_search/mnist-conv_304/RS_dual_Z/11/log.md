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
execution time: IAR + LP analysis = 14.79 + 33.80 = 48.59 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3551.41 seconds, max iter: 100)

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
rel_dist={2: [-0.9214935438597749, 0.9214933063013468]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=2.1853187084198
rel_dist={2: [-1.048994002949069, 1.0489933180675663]}

## Binary Search Result
Binary search time: 222.86 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Relational Split (RS_dual_Z) starts
Time budget: 3328.54 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 4616

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5710314, upper bound: 1.5739989
time: 8.18 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5739987, upper bound: 1.5710331
time: 9.01 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 17.45 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 17.45
Output dim: 2, lower bound: -1.5710314, upper bound: 1.5739989
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 17.45
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

Time for backsubstitution: 15.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 4656

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5709938, upper bound: 1.5566275
time: 7.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5536535, upper bound: 1.5739606
time: 5.37 seconds

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

Time for backsubstitution: 14.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 4656

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5739610, upper bound: 1.5536536
time: 4.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5566282, upper bound: 1.5709939
time: 5.09 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 25.08 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 25.08
Output dim: 2, lower bound: -1.5709938, upper bound: 1.5566275
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 25.08
Output dim: 2, lower bound: -1.5536535, upper bound: 1.5739606
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 25.08
Output dim: 2, lower bound: -1.5739610, upper bound: 1.5536536
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 25.08
Output dim: 2, lower bound: -1.5566282, upper bound: 1.5709939

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -3.0888672, 3.0986319
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.4943795, 2.4987788
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.5671983, 2.5627337
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -3.2390800, 3.2406778
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.4033775, 3.3940725
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.9585381, 2.9587941
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.7370157, 2.7381713
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.4790339, 3.4756398
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.2797709, 2.2857146
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -3.0142555, 3.0143499

Time for backsubstitution: 14.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 498

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5709771, upper bound: 1.5539974
time: 7.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5683633, upper bound: 1.5566127
time: 7.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -3.0932055, 3.0942936
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.4936776, 2.4994800
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.5621066, 2.5678258
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -3.2406778, 3.2390952
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.3967228, 3.4007277
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.9612818, 2.9560504
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.7408895, 2.7342973
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.4751925, 3.4794831
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.2856517, 2.2798333
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -3.0117350, 3.0168705

Time for backsubstitution: 14.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 498

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5536368, upper bound: 1.5713301
time: 5.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5510230, upper bound: 1.5739437
time: 5.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -3.0915136, 3.0932055
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.4968820, 2.4936783
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.5675073, 2.5621064
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -3.2390876, 3.2406778
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.4007282, 3.3953719
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.9560499, 2.9600258
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.7342978, 2.7394631
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.4792552, 3.4751916
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.2797809, 2.2856517
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -3.0155201, 3.0117350

Time for backsubstitution: 14.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 498

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5739442, upper bound: 1.5510225
time: 5.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5713305, upper bound: 1.5536369
time: 6.12 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -3.0958519, 3.0888672
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.4961810, 2.4943795
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.5624156, 2.5671983
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -3.2406778, 3.2390795
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.3940716, 3.4020276
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.9587946, 2.9572821
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.7381716, 2.7355890
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.4754119, 3.4790344
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.2856617, 2.2797709
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -3.0130005, 3.0142555

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 498

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5566109, upper bound: 1.5683635
time: 5.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5539977, upper bound: 1.5709774
time: 5.13 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 25.16 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.16
Output dim: 2, lower bound: -1.5709771, upper bound: 1.5539974
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.16
Output dim: 2, lower bound: -1.5683633, upper bound: 1.5566127
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.16
Output dim: 2, lower bound: -1.5536368, upper bound: 1.5713301
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.16
Output dim: 2, lower bound: -1.5510230, upper bound: 1.5739437
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.16
Output dim: 2, lower bound: -1.5739442, upper bound: 1.5510225
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.16
Output dim: 2, lower bound: -1.5713305, upper bound: 1.5536369
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.16
Output dim: 2, lower bound: -1.5566109, upper bound: 1.5683635
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.16
Output dim: 2, lower bound: -1.5539977, upper bound: 1.5709774

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -3.0934372, 3.0938888
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.4973030, 2.4957500
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.5679426, 2.5619578
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -3.2385373, 3.2406778
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.3966427, 3.4006224
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.9475555, 2.9693666
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.7327509, 2.7422748
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.4777493, 3.4768748
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.2835288, 2.2818370
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -3.0152426, 3.0133200

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 6219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5709696, upper bound: 1.5501014
time: 5.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5670810, upper bound: 1.5539902
time: 8.28 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -3.0841236, 3.0986319
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.4913502, 2.4987788
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.5664225, 2.5627337
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -3.2390800, 3.2406778
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.4033775, 3.3873363
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.9585381, 2.9478116
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.7370157, 2.7339065
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.4790339, 3.4743552
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.2758932, 2.2857146
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -3.0132256, 3.0143499

Time for backsubstitution: 14.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 6219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5683558, upper bound: 1.5527146
time: 4.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5644672, upper bound: 1.5566038
time: 6.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -3.0977745, 3.0895505
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.4966030, 2.4964511
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.5628510, 2.5670500
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -3.2406778, 3.2396202
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.3899860, 3.4072781
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.9502993, 2.9666224
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.7366247, 2.7384009
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.4739060, 3.4807181
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.2894096, 2.2759562
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -3.0127220, 3.0158405

Time for backsubstitution: 14.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 6219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5536293, upper bound: 1.5674343
time: 5.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5497407, upper bound: 1.5713229
time: 8.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -3.0884628, 3.0942936
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.4906502, 2.4994800
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.5613308, 2.5678258
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -3.2406778, 3.2385521
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.3967228, 3.3939915
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.9612818, 2.9450674
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.7408895, 2.7300324
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.4751925, 3.4781981
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.2817740, 2.2798333
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -3.0107059, 3.0168705

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 6219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5510156, upper bound: 1.5700498
time: 5.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5471270, upper bound: 1.5739366
time: 15.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -3.0960836, 3.0884619
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.4998055, 2.4906495
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.5682516, 2.5613306
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -3.2385449, 3.2406778
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.3939915, 3.4019217
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.9450674, 2.9705987
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.7300329, 2.7435665
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.4779706, 3.4764266
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.2835388, 2.2817740
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -3.0165071, 3.0107055

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 6219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5739367, upper bound: 1.5471287
time: 5.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5700481, upper bound: 1.5510151
time: 6.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -3.0867710, 3.0932055
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.4938526, 2.4936783
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.5667315, 2.5621064
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -3.2390876, 3.2406778
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.4007282, 3.3886356
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.9560499, 2.9490438
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.7342978, 2.7351983
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.4792552, 3.4739060
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.2759032, 2.2856517
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -3.0144901, 3.0117350

Time for backsubstitution: 14.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 6219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5713231, upper bound: 1.5497410
time: 5.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5674345, upper bound: 1.5536291
time: 5.19 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -3.1004219, 3.0841236
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.4991055, 2.4913507
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.5631599, 2.5664225
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -3.2406778, 3.2396045
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.3873367, 3.4085774
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.9478111, 2.9678545
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.7339067, 2.7396927
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.4741273, 3.4802694
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.2894197, 2.2758937
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -3.0139866, 3.0132260

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 6219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5566035, upper bound: 1.5644670
time: 5.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5527149, upper bound: 1.5683558
time: 6.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -3.0911083, 3.0888672
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.4931526, 2.4943795
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.5616398, 2.5671983
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -3.2406778, 3.2385364
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.3940716, 3.3952913
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.9587946, 2.9462996
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.7381716, 2.7313242
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.4754119, 3.4777493
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.2817841, 2.2797709
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -3.0119705, 3.0142555

Time for backsubstitution: 14.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 6219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5539902, upper bound: 1.5670808
time: 5.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5501016, upper bound: 1.5709694
time: 12.44 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 32.93 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 32.93
Output dim: 2, lower bound: -1.5709696, upper bound: 1.5501014
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 32.93
Output dim: 2, lower bound: -1.5670810, upper bound: 1.5539902
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 32.93
Output dim: 2, lower bound: -1.5683558, upper bound: 1.5527146
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 32.93
Output dim: 2, lower bound: -1.5644672, upper bound: 1.5566038
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 32.93
Output dim: 2, lower bound: -1.5536293, upper bound: 1.5674343
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 32.93
Output dim: 2, lower bound: -1.5497407, upper bound: 1.5713229
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 32.93
Output dim: 2, lower bound: -1.5510156, upper bound: 1.5700498
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 32.93
Output dim: 2, lower bound: -1.5471270, upper bound: 1.5739366
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 32.93
Output dim: 2, lower bound: -1.5739367, upper bound: 1.5471287
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 32.93
Output dim: 2, lower bound: -1.5700481, upper bound: 1.5510151
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 32.93
Output dim: 2, lower bound: -1.5713231, upper bound: 1.5497410
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 32.93
Output dim: 2, lower bound: -1.5674345, upper bound: 1.5536291
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 32.93
Output dim: 2, lower bound: -1.5566035, upper bound: 1.5644670
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 32.93
Output dim: 2, lower bound: -1.5527149, upper bound: 1.5683558
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 32.93
Output dim: 2, lower bound: -1.5539902, upper bound: 1.5670808
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 32.93
Output dim: 2, lower bound: -1.5501016, upper bound: 1.5709694

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -3.0763330, 3.0817747
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.5014148, 2.5020885
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.5613608, 2.5486617
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -3.2406778, 3.2406778
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.3851929, 3.3925066
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.9519091, 2.9760828
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.7048683, 2.7225258
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.4812675, 3.4822979
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.2588315, 2.2643552
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -3.0168161, 3.0143409

Time for backsubstitution: 14.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 6170

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5709680, upper bound: 1.5447339
time: 5.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5656086, upper bound: 1.5500993
time: 4.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -3.0813227, 3.0767851
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.5036416, 2.4998615
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.5546465, 2.5553763
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -3.2406778, 3.2406778
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.3885250, 3.3891745
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.9542713, 2.9737201
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.7130013, 2.7143934
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.4831729, 3.4803925
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.2660475, 2.2571397
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -3.0162630, 3.0148940

Time for backsubstitution: 14.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 6170

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5670794, upper bound: 1.5486240
time: 5.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5617197, upper bound: 1.5539886
time: 4.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -3.0670195, 3.0865178
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.4954619, 2.5051174
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.5598407, 2.5494380
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -3.2406778, 3.2406778
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.3919296, 3.3792205
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.9628916, 2.9545279
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.7091341, 2.7141576
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.4825511, 3.4797778
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.2511959, 2.2682323
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -3.0147991, 3.0153704

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 6170

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5683542, upper bound: 1.5473481
time: 5.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5629939, upper bound: 1.5527125
time: 4.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -3.0720091, 3.0815282
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.4976888, 2.5028903
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.5531263, 2.5561526
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -3.2406778, 3.2406778
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.3952618, 3.3758883
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.9652538, 2.9521651
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.7172651, 2.7060251
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.4844565, 3.4778724
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.2584119, 2.2610164
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -3.0142460, 3.0159235

Time for backsubstitution: 14.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 6170

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5644656, upper bound: 1.5512383
time: 5.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5591050, upper bound: 1.5566013
time: 4.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -3.0806704, 3.0774364
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.5007138, 2.5027895
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.5562692, 2.5537539
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -3.2406778, 3.2406778
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.3785381, 3.3991623
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.9546537, 2.9733386
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.7087431, 2.7186518
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.4774241, 3.4861412
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.2647123, 2.2584743
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -3.0142956, 3.0168614

Time for backsubstitution: 14.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 6170

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5536278, upper bound: 1.5620707
time: 5.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5482659, upper bound: 1.5674322
time: 4.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -3.0856600, 3.0724473
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.5029407, 2.5005627
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.5495543, 2.5604682
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -3.2406778, 3.2406778
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.3818703, 3.3958302
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.9570160, 2.9709759
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.7168751, 2.7105193
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.4793296, 3.4842353
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.2719283, 2.2512584
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -3.0137424, 3.0174146

Time for backsubstitution: 14.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 6170

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5497391, upper bound: 1.5659593
time: 5.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5443755, upper bound: 1.5713209
time: 4.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -3.0713587, 3.0821795
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.4947610, 2.5058184
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.5547490, 2.5545301
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -3.2406778, 3.2406778
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.3852749, 3.3858757
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.9656353, 2.9517837
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.7130079, 2.7102835
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.4787097, 3.4836206
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.2570767, 2.2623515
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -3.0122795, 3.0178909

Time for backsubstitution: 14.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 6170

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5510140, upper bound: 1.5646853
time: 5.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5456511, upper bound: 1.5700455
time: 5.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -3.0763483, 3.0771899
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.4969878, 2.5035915
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.5480342, 2.5612445
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -3.2406778, 3.2406778
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.3886070, 3.3825436
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.9679985, 2.9494214
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.7211390, 2.7021513
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.4806151, 3.4817152
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.2642927, 2.2551355
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -3.0117264, 3.0184441

Time for backsubstitution: 14.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 6170

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5471254, upper bound: 1.5685738
time: 4.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5417607, upper bound: 1.5739345
time: 5.28 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -3.0789785, 3.0763478
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.5039182, 2.4969878
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.5616708, 2.5480344
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -3.2406778, 3.2406778
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.3825436, 3.3938065
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.9494209, 2.9773140
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.7021513, 2.7238176
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.4814868, 3.4818492
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.2588420, 2.2642927
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -3.0180807, 3.0117259

Time for backsubstitution: 14.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 6170

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5739345, upper bound: 1.5417608
time: 8.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5685742, upper bound: 1.5471249
time: 5.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -3.0839682, 3.0713587
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.5061450, 2.4947610
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.5549564, 2.5547490
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -3.2406778, 3.2406778
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.3858757, 3.3904743
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.9517832, 2.9749517
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.7102833, 2.7156851
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.4833922, 3.4799438
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.2660580, 2.2570767
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -3.0175276, 3.0122790

Time for backsubstitution: 15.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 6170

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5700459, upper bound: 1.5456511
time: 7.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5646856, upper bound: 1.5510136
time: 5.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -3.0696659, 3.0810909
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.4979653, 2.5000167
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.5601506, 2.5488107
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -3.2406778, 3.2406778
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.3892784, 3.3805199
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.9604034, 2.9557590
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.7064161, 2.7154493
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.4827724, 3.4793291
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.2512064, 2.2681699
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -3.0160637, 3.0127554

Time for backsubstitution: 14.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 6170

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5713209, upper bound: 1.5443758
time: 5.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5659596, upper bound: 1.5497388
time: 5.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -3.0746546, 3.0761018
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.5001922, 2.4977899
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.5534363, 2.5555253
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -3.2406778, 3.2406778
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.3926105, 3.3771877
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.9627666, 2.9533968
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.7145472, 2.7073169
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.4846778, 3.4774237
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.2584224, 2.2609539
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -3.0155106, 3.0133090

Time for backsubstitution: 14.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 6170

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5674323, upper bound: 1.5482661
time: 5.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5620711, upper bound: 1.5536274
time: 5.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -3.0833168, 3.0720096
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.5032172, 2.4976892
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.5565791, 2.5531263
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -3.2406778, 3.2406778
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.3758869, 3.4004617
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.9521656, 2.9745698
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.7060251, 2.7199435
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.4776454, 3.4856925
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.2647228, 2.2584119
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -3.0155602, 3.0142465

Time for backsubstitution: 14.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 6170

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5566013, upper bound: 1.5591046
time: 5.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5512385, upper bound: 1.5644659
time: 5.36 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 25.95 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.95
Output dim: 2, lower bound: -1.5709680, upper bound: 1.5447339
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.95
Output dim: 2, lower bound: -1.5656086, upper bound: 1.5500993
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.95
Output dim: 2, lower bound: -1.5670794, upper bound: 1.5486240
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.95
Output dim: 2, lower bound: -1.5617197, upper bound: 1.5539886
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.95
Output dim: 2, lower bound: -1.5683542, upper bound: 1.5473481
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.95
Output dim: 2, lower bound: -1.5629939, upper bound: 1.5527125
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.95
Output dim: 2, lower bound: -1.5644656, upper bound: 1.5512383
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.95
Output dim: 2, lower bound: -1.5591050, upper bound: 1.5566013
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.95
Output dim: 2, lower bound: -1.5536278, upper bound: 1.5620707
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.95
Output dim: 2, lower bound: -1.5482659, upper bound: 1.5674322
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.95
Output dim: 2, lower bound: -1.5497391, upper bound: 1.5659593
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.95
Output dim: 2, lower bound: -1.5443755, upper bound: 1.5713209
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.95
Output dim: 2, lower bound: -1.5510140, upper bound: 1.5646853
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.95
Output dim: 2, lower bound: -1.5456511, upper bound: 1.5700455
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.95
Output dim: 2, lower bound: -1.5471254, upper bound: 1.5685738
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.95
Output dim: 2, lower bound: -1.5417607, upper bound: 1.5739345
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.95
Output dim: 2, lower bound: -1.5739345, upper bound: 1.5417608
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.95
Output dim: 2, lower bound: -1.5685742, upper bound: 1.5471249
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.95
Output dim: 2, lower bound: -1.5700459, upper bound: 1.5456511
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.95
Output dim: 2, lower bound: -1.5646856, upper bound: 1.5510136
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.95
Output dim: 2, lower bound: -1.5713209, upper bound: 1.5443758
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.95
Output dim: 2, lower bound: -1.5659596, upper bound: 1.5497388
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.95
Output dim: 2, lower bound: -1.5674323, upper bound: 1.5482661
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.95
Output dim: 2, lower bound: -1.5620711, upper bound: 1.5536274
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.95
Output dim: 2, lower bound: -1.5566013, upper bound: 1.5591046
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.95
Output dim: 2, lower bound: -1.5512385, upper bound: 1.5644659
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.95
Output dim: 2, lower bound: -1.5527149, upper bound: 1.5683558
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.95
Output dim: 2, lower bound: -1.5539902, upper bound: 1.5670808
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.95
Output dim: 2, lower bound: -1.5501016, upper bound: 1.5709694
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=2.579909086227417
rel_dist={2: [-1.574020518019939, 1.5740203796460204]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 4616

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2820899, upper bound: 1.2849159
time: 5.98 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2849161, upper bound: 1.2820896
time: 6.80 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 13.02 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 13.02
Output dim: 2, lower bound: -1.2820899, upper bound: 1.2849159
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 13.02
Output dim: 2, lower bound: -1.2849161, upper bound: 1.2820896

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.8282852, 2.8313856
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2816811, 2.2845960
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.3428454, 2.3432040
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -3.0074120, 3.0074210
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.0937281, 3.0922141
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.6177568, 2.6163359
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.4303484, 2.4287953
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.3074045, 3.3076601
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.1260943, 2.1261301
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7868681, 2.7883625

Time for backsubstitution: 15.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 4656

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2820861, upper bound: 1.2743951
time: 12.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2715743, upper bound: 1.2849125
time: 9.11 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.8309317, 2.8282847
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2841835, 2.2816815
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.3431549, 2.3428454
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -3.0074196, 3.0074120
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.0922136, 3.0935130
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.6163359, 2.6175675
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.4287958, 2.4300876
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.3076239, 3.3074040
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.1261039, 2.1260943
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7881327, 2.7868681

Time for backsubstitution: 14.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 4656

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2849124, upper bound: 1.2715743
time: 10.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2743930, upper bound: 1.2820858
time: 5.84 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 31.42 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 31.42
Output dim: 2, lower bound: -1.2820861, upper bound: 1.2743951
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 31.42
Output dim: 2, lower bound: -1.2715743, upper bound: 1.2849125
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 31.42
Output dim: 2, lower bound: -1.2849124, upper bound: 1.2715743
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 31.42
Output dim: 2, lower bound: -1.2743930, upper bound: 1.2820858

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.8133640, 2.8189440
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2820134, 2.2845275
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.3282619, 2.3257108
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9910550, 2.9937806
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.0746193, 3.0693016
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.6083088, 2.6084552
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.4170027, 2.4176641
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.3106151, 3.3086758
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.1059036, 2.1093001
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7919941, 2.7920475

Time for backsubstitution: 14.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 498

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2820716, upper bound: 1.2718092
time: 8.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2794963, upper bound: 1.2743800
time: 6.46 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.8158426, 2.8164649
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2816129, 2.2849283
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.3253522, 2.3286204
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9937711, 2.9910645
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.0708160, 3.0731049
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.6098766, 2.6068873
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.4192171, 2.4154501
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.3084197, 3.3108721
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.1092644, 2.1059394
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7905531, 2.7934875

Time for backsubstitution: 14.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 498

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2715594, upper bound: 1.2823197
time: 8.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2689901, upper bound: 1.2848973
time: 10.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.8160105, 2.8158426
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2845159, 2.2816131
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.3285708, 2.3253522
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9910626, 2.9937711
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.0731049, 3.0706015
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.6068869, 2.6096869
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.4154501, 2.4189558
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.3108363, 3.3084192
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.1059136, 2.1092644
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7932587, 2.7905531

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 498

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2848972, upper bound: 1.2689905
time: 15.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2823201, upper bound: 1.2715594
time: 6.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.8184900, 2.8133640
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2841153, 2.2820137
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.3256612, 2.3282619
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9937787, 2.9910555
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.0693016, 3.0744047
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.6084547, 2.6081190
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.4176645, 2.4167418
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.3086410, 3.3106155
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.1092744, 2.1059041
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7918186, 2.7919936

Time for backsubstitution: 14.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 498

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2743780, upper bound: 1.2794961
time: 5.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2718089, upper bound: 1.2820711
time: 5.56 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 26.23 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.23
Output dim: 2, lower bound: -1.2820716, upper bound: 1.2718092
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 26.23
Output dim: 2, lower bound: -1.2794963, upper bound: 1.2743800
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.23
Output dim: 2, lower bound: -1.2715594, upper bound: 1.2823197
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 26.23
Output dim: 2, lower bound: -1.2689901, upper bound: 1.2848973
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.23
Output dim: 2, lower bound: -1.2848972, upper bound: 1.2689905
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 26.23
Output dim: 2, lower bound: -1.2823201, upper bound: 1.2715594
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.23
Output dim: 2, lower bound: -1.2743780, upper bound: 1.2794961
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 26.23
Output dim: 2, lower bound: -1.2718089, upper bound: 1.2820711

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.8139420, 2.8142004
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2823868, 2.2814987
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.3283548, 2.3249350
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9905124, 2.9938478
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.0678825, 3.0701575
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.5973263, 2.6097898
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.4127378, 2.4181809
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.3093305, 3.3088307
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.1063895, 2.1054225
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7921162, 2.7910180

Time for backsubstitution: 14.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 6219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2820677, upper bound: 1.2696050
time: 9.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2798480, upper bound: 1.2718039
time: 6.48 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.8086205, 2.8189440
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2789841, 2.2845275
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.3274860, 2.3257108
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9910550, 2.9932370
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.0746193, 3.0625653
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.6083088, 2.5974727
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.4170027, 2.4133992
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.3106151, 3.3073907
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.1020265, 2.1093001
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7909641, 2.7920475

Time for backsubstitution: 14.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 6219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2794923, upper bound: 1.2721696
time: 9.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2772874, upper bound: 1.2743739
time: 45.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.8164215, 2.8117218
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2819862, 2.2818995
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.3254452, 2.3278446
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9932275, 2.9911318
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.0640793, 3.0739608
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.5988941, 2.6082220
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.4149523, 2.4159675
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.3071342, 3.3110266
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.1097498, 2.1020622
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7906761, 2.7924581

Time for backsubstitution: 14.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 6219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2715555, upper bound: 1.2801151
time: 5.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2693416, upper bound: 1.2823158
time: 13.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.8111000, 2.8164649
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2785835, 2.2849283
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.3245764, 2.3286204
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9937711, 2.9905214
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.0708160, 3.0663686
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.6098766, 2.5959044
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.4192171, 2.4111857
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.3084197, 3.3095865
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.1053867, 2.1059394
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7895241, 2.7934875

Time for backsubstitution: 14.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 6219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2689861, upper bound: 1.2826784
time: 5.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2667816, upper bound: 1.2848953
time: 9.86 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.8165894, 2.8110995
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2848892, 2.2785842
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.3286638, 2.3245764
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9905200, 2.9938383
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.0663681, 3.0714574
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.5959044, 2.6110220
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.4111853, 2.4194727
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.3095508, 3.3085747
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.1063995, 2.1053867
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7933807, 2.7895236

Time for backsubstitution: 15.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 6219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2848933, upper bound: 1.2667807
time: 5.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2826788, upper bound: 1.2689859
time: 6.23 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.8112679, 2.8158426
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2814865, 2.2816131
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.3277950, 2.3253522
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9910626, 2.9932280
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.0731049, 3.0638652
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.6068869, 2.5987048
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.4154501, 2.4146910
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.3108363, 3.3071346
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.1020360, 2.1092644
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7922287, 2.7905531

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 6219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2823161, upper bound: 1.2693413
time: 5.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2801157, upper bound: 1.2715553
time: 6.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.8190680, 2.8086205
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2844887, 2.2789848
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.3257542, 2.3274860
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9932351, 2.9911227
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.0625648, 3.0752606
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.5974722, 2.6094537
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.4133997, 2.4172592
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.3073554, 3.3107700
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.1097598, 2.1020265
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7919407, 2.7909641

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 6219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2743741, upper bound: 1.2772872
time: 6.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2721676, upper bound: 1.2794915
time: 5.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.8137465, 2.8133640
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2810860, 2.2820137
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.3248854, 2.3282619
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9937787, 2.9905124
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.0693016, 3.0676684
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.6084547, 2.5971370
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.4176645, 2.4124775
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.3086410, 3.3093300
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.1053967, 2.1059041
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7907887, 2.7919936

Time for backsubstitution: 14.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 6219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2718049, upper bound: 1.2798477
time: 5.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2696050, upper bound: 1.2820666
time: 5.55 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 26.29 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.29
Output dim: 2, lower bound: -1.2820677, upper bound: 1.2696050
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.29
Output dim: 2, lower bound: -1.2798480, upper bound: 1.2718039
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.29
Output dim: 2, lower bound: -1.2794923, upper bound: 1.2721696
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.29
Output dim: 2, lower bound: -1.2772874, upper bound: 1.2743739
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.29
Output dim: 2, lower bound: -1.2715555, upper bound: 1.2801151
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.29
Output dim: 2, lower bound: -1.2693416, upper bound: 1.2823158
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.29
Output dim: 2, lower bound: -1.2689861, upper bound: 1.2826784
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.29
Output dim: 2, lower bound: -1.2667816, upper bound: 1.2848953
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.29
Output dim: 2, lower bound: -1.2848933, upper bound: 1.2667807
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.29
Output dim: 2, lower bound: -1.2826788, upper bound: 1.2689859
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.29
Output dim: 2, lower bound: -1.2823161, upper bound: 1.2693413
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.29
Output dim: 2, lower bound: -1.2801157, upper bound: 1.2715553
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.29
Output dim: 2, lower bound: -1.2743741, upper bound: 1.2772872
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.29
Output dim: 2, lower bound: -1.2721676, upper bound: 1.2794915
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.29
Output dim: 2, lower bound: -1.2718049, upper bound: 1.2798477
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.29
Output dim: 2, lower bound: -1.2696050, upper bound: 1.2820666

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.7968378, 2.7999482
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2864976, 2.2868829
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.3188953, 2.3116388
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9936118, 2.9979086
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.0564346, 3.0606136
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.6016798, 2.6154933
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.3848572, 2.3949466
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.3128486, 3.3134370
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.0816922, 2.0848484
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7934532, 2.7920384

Time for backsubstitution: 14.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 6170

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2820653, upper bound: 1.2666594
time: 13.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2791213, upper bound: 1.2696022
time: 9.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.7996893, 2.7970972
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2877698, 2.2856102
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.3150582, 2.3154755
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9945731, 2.9969463
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.0583382, 3.0587101
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.6030302, 2.6141438
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.3895044, 2.3902998
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.3139358, 3.3123479
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.0858154, 2.0807252
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7931366, 2.7923546

Time for backsubstitution: 15.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 6170

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2798456, upper bound: 1.2688572
time: 5.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2638432, upper bound: 1.2718018
time: 10.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.7915163, 2.8046913
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2830958, 2.2899117
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.3180265, 2.3124151
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9941554, 2.9972982
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.0631714, 3.0530214
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.6126623, 2.6031766
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.3891211, 2.3901649
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.3141322, 3.3119969
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.0773287, 2.0887256
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7923012, 2.7930679

Time for backsubstitution: 14.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 6170

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2794900, upper bound: 1.2692181
time: 7.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2765448, upper bound: 1.2721647
time: 8.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.7943678, 2.8018403
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2843690, 2.2886391
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.3141899, 2.3162518
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9951167, 2.9963360
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.0650749, 3.0511179
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.6140127, 2.6018262
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.3937693, 2.3855176
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.3152213, 3.3109078
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.0814524, 2.0846024
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7919846, 2.7933841

Time for backsubstitution: 14.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 6170

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2772850, upper bound: 1.2714263
time: 5.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2743433, upper bound: 1.2743709
time: 8.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.7993174, 2.7974691
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2860970, 2.2872834
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.3159857, 2.3145485
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9963269, 2.9951930
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.0526314, 3.0644169
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.6032476, 2.6139255
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.3870707, 2.3927331
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.3106513, 3.3156333
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.0850525, 2.0814881
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7920132, 2.7934785

Time for backsubstitution: 14.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 6170

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2715532, upper bound: 1.2771674
time: 5.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2686107, upper bound: 1.2801125
time: 5.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.8021688, 2.7946181
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2873693, 2.2860110
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.3121486, 2.3183851
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9972892, 2.9942307
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.0545349, 3.0625129
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.6045980, 2.6125755
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.3917179, 2.3880858
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.3117404, 3.3145442
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.0891757, 2.0773644
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7916965, 2.7937951

Time for backsubstitution: 14.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 6170

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2693392, upper bound: 1.2793702
time: 22.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2664001, upper bound: 1.2823127
time: 5.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.7939959, 2.8022122
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2826953, 2.2903123
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.3151174, 2.3153248
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9968705, 2.9945827
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.0593681, 3.0568247
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.6142311, 2.6016083
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.3913355, 2.3879514
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.3119369, 3.3141932
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.0806894, 2.0853653
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7908611, 2.7945085

Time for backsubstitution: 14.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 6170

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2689837, upper bound: 1.2797404
time: 6.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2660399, upper bound: 1.2826752
time: 5.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.7968473, 2.7993612
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2839684, 2.2890399
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.3112803, 2.3191614
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9978328, 2.9936204
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.0612717, 3.0549212
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.6155806, 2.6002584
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.3959818, 2.3833041
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.3130260, 3.3131042
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.0848126, 2.0812416
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7905445, 2.7948241

Time for backsubstitution: 14.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 6170

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2667787, upper bound: 1.2819436
time: 6.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2638416, upper bound: 1.2848905
time: 8.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.7994843, 2.7968469
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2890010, 2.2839682
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.3192053, 2.3112803
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9936194, 2.9979000
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.0549202, 3.0619135
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.6002588, 2.6167250
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.3833046, 2.3962383
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.3130679, 3.3131804
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.0817027, 2.0848126
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7947178, 2.7905440

Time for backsubstitution: 15.00 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=2.3431549072265625
rel_dist={2: [-1.2849368558531555, 1.2849364906580067]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 4616

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1664042, upper bound: 1.1688883
time: 14.54 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1688884, upper bound: 1.1664040
time: 5.69 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 20.48 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 20.48
Output dim: 2, lower bound: -1.1664042, upper bound: 1.1688883
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 20.48
Output dim: 2, lower bound: -1.1688884, upper bound: 1.1664040

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.7364502, 2.7387762
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2109928, 2.2131789
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2639270, 2.2641962
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9247379, 2.9247437
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -2.9850931, 2.9839573
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.5010138, 2.4999480
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.3236780, 2.3225131
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.2518129, 3.2520051
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.0681386, 2.0681653
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7131414, 2.7142620

Time for backsubstitution: 14.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 4656

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -1.1664005, upper bound: 1.1609287
time: 10.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1584422, upper bound: 1.1688847
time: 8.10 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.7387762, 2.7364502
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2131786, 2.2109931
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2641959, 2.2639272
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9247437, 2.9247370
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -2.9839563, 2.9850931
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.4999475, 2.5010142
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.3225126, 2.3236775
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.2520056, 3.2518129
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.0681653, 2.0681386
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7142611, 2.7131410

Time for backsubstitution: 14.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4656
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 4656

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1688848, upper bound: 1.1584421
time: 20.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -1.1609264, upper bound: 1.1664006
time: 7.21 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 42.57 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 42.57
Output dim: 2, lower bound: -1.1664005, upper bound: 1.1609287
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 42.57
Output dim: 2, lower bound: -1.1584422, upper bound: 1.1688847
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 42.57
Output dim: 2, lower bound: -1.1688848, upper bound: 1.1584421
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 42.57
Output dim: 2, lower bound: -1.1609264, upper bound: 1.1664006

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.7233887, 2.7238550
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2109246, 2.2134111
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2464337, 2.2488852
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9104180, 2.9083877
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -2.9621811, 2.9638977
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.4927416, 2.4905000
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.3119936, 2.3091679
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.2528281, 3.2546682
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.0504684, 2.0479751
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7168255, 2.7190270

Time for backsubstitution: 14.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 498

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1584373, upper bound: 1.1667466
time: 13.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1563045, upper bound: 1.1688798
time: 13.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.7238550, 2.7233887
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2134109, 2.2109246
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2488852, 2.2464340
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9083877, 2.9104180
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -2.9638977, 2.9621811
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.4904995, 2.4927421
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.3091688, 2.3119931
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.2546687, 3.2528286
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.0479751, 2.0504684
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7190266, 2.7168260

Time for backsubstitution: 14.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 498

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1688798, upper bound: 1.1563039
time: 5.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1667468, upper bound: 1.1584393
time: 10.35 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 31.15 seconds
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 31.15
Output dim: 2, lower bound: -1.1584373, upper bound: 1.1667466
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 31.15
Output dim: 2, lower bound: -1.1563045, upper bound: 1.1688798
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 31.15
Output dim: 2, lower bound: -1.1688798, upper bound: 1.1563039
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 31.15
Output dim: 2, lower bound: -1.1667468, upper bound: 1.1584393

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.7226372, 2.7191119
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2104473, 2.2103822
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2463098, 2.2481093
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9098744, 2.9083023
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -2.9554443, 2.9628553
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.4817591, 2.4887547
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.3077288, 2.3084893
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.2515435, 3.2544632
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.0498633, 2.0440974
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7166605, 2.7179976

Time for backsubstitution: 14.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 6219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -1.1584343, upper bound: 1.1651201
time: 17.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1568119, upper bound: 1.1667435
time: 6.04 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.7186451, 2.7231030
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2078953, 2.2129335
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2456579, 2.2487609
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9103322, 2.9078445
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -2.9611378, 2.9571614
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.4909973, 2.4795170
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.3113146, 2.3049030
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.2526231, 3.2533827
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.0465908, 2.0473695
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7157965, 2.7188616

Time for backsubstitution: 14.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 6219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1563015, upper bound: 1.1672528
time: 8.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1546791, upper bound: 1.1688765
time: 6.14 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.7231035, 2.7186451
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2129335, 2.2078958
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2487612, 2.2456582
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9078450, 2.9103322
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -2.9571609, 2.9611392
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.4795170, 2.4909973
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.3049040, 2.3113146
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.2533822, 3.2526236
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.0473695, 2.0465908
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7188616, 2.7157965

Time for backsubstitution: 14.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 6219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1688769, upper bound: 1.1546790
time: 9.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1672529, upper bound: 1.1563011
time: 7.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.7191124, 2.7226367
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2103825, 2.2104471
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2481093, 2.2463098
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9083028, 2.9098744
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -2.9628544, 2.9554448
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.4887543, 2.4817591
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.3084898, 2.3077283
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.2544627, 3.2515435
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.0440974, 2.0498633
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7179976, 2.7166605

Time for backsubstitution: 14.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 6219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1667439, upper bound: 1.1568118
time: 14.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -1.1651199, upper bound: 1.1584338
time: 5.38 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 34.97 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 34.97
Output dim: 2, lower bound: -1.1584343, upper bound: 1.1651201
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 34.97
Output dim: 2, lower bound: -1.1568119, upper bound: 1.1667435
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 34.97
Output dim: 2, lower bound: -1.1563015, upper bound: 1.1672528
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 34.97
Output dim: 2, lower bound: -1.1546791, upper bound: 1.1688765
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 34.97
Output dim: 2, lower bound: -1.1688769, upper bound: 1.1546790
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 34.97
Output dim: 2, lower bound: -1.1672529, upper bound: 1.1563011
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 34.97
Output dim: 2, lower bound: -1.1667439, upper bound: 1.1568118
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 34.97
Output dim: 2, lower bound: -1.1651199, upper bound: 1.1584338

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.7076712, 2.7020087
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2155128, 2.2144938
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2330136, 2.2376909
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9136949, 2.9114013
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -2.9454250, 2.9514074
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.4871254, 2.4931083
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.2833319, 2.2806082
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.2558780, 3.2579803
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.0282583, 2.0194001
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7176819, 2.7192550

Time for backsubstitution: 14.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 6170

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -1.1568099, upper bound: 1.1644306
time: 11.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1544967, upper bound: 1.1667409
time: 5.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.7015409, 2.7081380
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2120070, 2.2179995
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2352395, 2.2354648
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9134307, 2.9116650
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -2.9496899, 2.9471412
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.4953508, 2.4848833
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.2834330, 2.2805071
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.2561412, 3.2577167
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.0218935, 2.0257649
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7170544, 2.7198820

Time for backsubstitution: 14.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 6170

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -1.1562995, upper bound: 1.1649359
time: 17.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1539877, upper bound: 1.1672495
time: 5.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.7036810, 2.7059999
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2129617, 2.2170451
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2323623, 2.2383423
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9141526, 2.9109430
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -2.9511185, 2.9457130
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.4963636, 2.4838710
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.2869186, 2.2770219
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.2569575, 3.2569003
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.0249858, 2.0226722
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7168179, 2.7201190

Time for backsubstitution: 15.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 6170

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1546771, upper bound: 1.1665632
time: 8.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1523638, upper bound: 1.1688745
time: 5.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.7060003, 2.7036805
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2170453, 2.2129617
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2383423, 2.2323620
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9109435, 2.9141531
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -2.9457130, 2.9511194
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.4838705, 2.4963632
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.2770214, 2.2869186
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.2569003, 3.2569575
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.0226722, 2.0249858
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7201195, 2.7168169

Time for backsubstitution: 14.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 6170

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1688744, upper bound: 1.1523637
time: 5.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1665636, upper bound: 1.1546768
time: 13.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.7081385, 2.7015419
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2179990, 2.2120073
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2354646, 2.2352397
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9116654, 2.9134312
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -2.9471397, 2.9496908
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.4848833, 2.4953508
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.2805071, 2.2834334
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.2577167, 3.2561412
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.0257649, 2.0218935
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7198830, 2.7170539

Time for backsubstitution: 14.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 6170

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1672505, upper bound: 1.1539875
time: 6.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -1.1649360, upper bound: 1.1562992
time: 9.10 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.7020082, 2.7076716
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2144933, 2.2155130
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2376909, 2.2330136
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.9114013, 2.9136953
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -2.9514065, 2.9454250
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.4931087, 2.4871259
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.2806082, 2.2833323
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.2579799, 3.2558780
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.0194001, 2.0282583
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7192554, 2.7176814

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6170
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 6170

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1667415, upper bound: 1.1544989
time: 12.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -1.1644306, upper bound: 1.1568097
time: 6.20 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 33.49 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 33.49
Output dim: 2, lower bound: -1.1568099, upper bound: 1.1644306
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 33.49
Output dim: 2, lower bound: -1.1544967, upper bound: 1.1667409
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 33.49
Output dim: 2, lower bound: -1.1562995, upper bound: 1.1649359
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 33.49
Output dim: 2, lower bound: -1.1539877, upper bound: 1.1672495
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 33.49
Output dim: 2, lower bound: -1.1546771, upper bound: 1.1665632
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 33.49
Output dim: 2, lower bound: -1.1523638, upper bound: 1.1688745
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 33.49
Output dim: 2, lower bound: -1.1688744, upper bound: 1.1523637
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 33.49
Output dim: 2, lower bound: -1.1665636, upper bound: 1.1546768
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 33.49
Output dim: 2, lower bound: -1.1672505, upper bound: 1.1539875
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 33.49
Output dim: 2, lower bound: -1.1649360, upper bound: 1.1562992
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 33.49
Output dim: 2, lower bound: -1.1667415, upper bound: 1.1544989
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 33.49
Output dim: 2, lower bound: -1.1644306, upper bound: 1.1568097

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.7014856, 2.6929789
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2047386, 2.2021799
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2203045, 2.2270806
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.8963079, 2.8896070
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -2.9404044, 2.9456735
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.4672308, 2.4703736
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.2588472, 2.2488251
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.2436295, 3.2439837
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.0058093, 1.9997492
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7032576, 2.7066331

Time for backsubstitution: 14.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 4639

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -1.1544689, upper bound: 1.1607630
time: 11.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1485188, upper bound: 1.1667132
time: 5.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.6953564, 2.6991086
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2012329, 2.2056856
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2225308, 2.2248545
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.8960447, 2.8898706
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -2.9446702, 2.9414072
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.4754562, 2.4621482
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.2589483, 2.2487240
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.2438927, 3.2437205
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -1.9994445, 2.0061140
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7026310, 2.7072606

Time for backsubstitution: 14.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 4639

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -1.1539599, upper bound: 1.1612725
time: 5.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1480095, upper bound: 1.1672221
time: 5.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.6946507, 2.6998134
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2006483, 2.2062700
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2217541, 2.2256331
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.8923578, 2.8935566
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -2.9453845, 2.9406929
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.4736290, 2.4639788
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.2551355, 2.2525449
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.2429619, 3.2446523
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.0053382, 2.0002232
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7041979, 2.7056956

Time for backsubstitution: 14.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 4639

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -1.1546493, upper bound: 1.1605852
time: 5.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -1.1486988, upper bound: 1.1665356
time: 5.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.6974955, 2.6969705
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2021875, 2.2047312
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2196527, 2.2277322
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.8967657, 2.8891487
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -2.9460988, 2.9399791
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.4764690, 2.4611359
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.2624331, 2.2452388
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.2447090, 3.2429042
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.0025368, 2.0030212
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7023935, 2.7074971

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 4639

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -1.1523360, upper bound: 1.1628967
time: 7.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1463859, upper bound: 1.1688463
time: 14.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.6969709, 2.6974950
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2047310, 2.2021871
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2277322, 2.2196529
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.8891487, 2.8967662
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -2.9399781, 2.9460988
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.4611359, 2.4764686
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.2452383, 2.2624335
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.2429047, 3.2447095
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.0030212, 2.0025368
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7074976, 2.7023940

Time for backsubstitution: 14.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 4639

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1688465, upper bound: 1.1463859
time: 5.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -1.1628964, upper bound: 1.1523361
time: 6.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.6998129, 2.6946507
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2062702, 2.2006478
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2256331, 2.2217541
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.8935566, 2.8923583
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -2.9406924, 2.9453850
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.4639788, 2.4736285
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.2525454, 2.2551355
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.2446518, 3.2429614
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.0002232, 2.0053382
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7056952, 2.7041974

Time for backsubstitution: 14.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.25 seconds

### Candidate
type: RSZ, layer: 1, pos: 4639

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -1.1665356, upper bound: 1.1487010
time: 8.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -1.1605855, upper bound: 1.1546494
time: 19.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0259066, -5.5622492, -9.0259066, -5.5622492, -2.6991091, 2.6953564
1: -6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.2056856, 2.2012327
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2248545, 2.2225306
3: -6.1232843, -2.8826065, -6.1232843, -2.8826065, -2.8898706, 2.8960443
4: -11.8333845, -7.9824409, -11.8333845, -7.9824409, -2.9414067, 2.9446712
5: -13.6636562, -10.1825514, -13.6636562, -10.1825514, -2.4621487, 2.4754558
6: -15.6556635, -12.3171921, -15.6556635, -12.3171921, -2.2487249, 2.2589488
7: -5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.2437210, 3.2438927
8: -1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.0061140, 1.9994445
9: -7.3109250, -4.0054374, -7.3109250, -4.0054374, -2.7072611, 2.7026310

Time for backsubstitution: 14.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 5843
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 6191
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6231
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 4639

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1672225, upper bound: 1.1480095
time: 19.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -1.1612724, upper bound: 1.1539599
time: 19.67 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 54.58 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 54.58
Output dim: 2, lower bound: -1.1544689, upper bound: 1.1607630
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 54.58
Output dim: 2, lower bound: -1.1485188, upper bound: 1.1667132
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 54.58
Output dim: 2, lower bound: -1.1539599, upper bound: 1.1612725
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 54.58
Output dim: 2, lower bound: -1.1480095, upper bound: 1.1672221
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 54.58
Output dim: 2, lower bound: -1.1546493, upper bound: 1.1605852
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 54.58
Output dim: 2, lower bound: -1.1486988, upper bound: 1.1665356
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 54.58
Output dim: 2, lower bound: -1.1523360, upper bound: 1.1628967
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 54.58
Output dim: 2, lower bound: -1.1463859, upper bound: 1.1688463
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 54.58
Output dim: 2, lower bound: -1.1688465, upper bound: 1.1463859
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 54.58
Output dim: 2, lower bound: -1.1628964, upper bound: 1.1523361
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 54.58
Output dim: 2, lower bound: -1.1665356, upper bound: 1.1487010
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 54.58
Output dim: 2, lower bound: -1.1605855, upper bound: 1.1546494
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 54.58
Output dim: 2, lower bound: -1.1672225, upper bound: 1.1480095
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 54.58
Output dim: 2, lower bound: -1.1612724, upper bound: 1.1539599
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 54.58
Output dim: 2, lower bound: -1.1667415, upper bound: 1.1544989
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=2.2642369270324707
rel_dist={2: [-1.168894797061638, 1.1688945587998152]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2414.45 seconds
