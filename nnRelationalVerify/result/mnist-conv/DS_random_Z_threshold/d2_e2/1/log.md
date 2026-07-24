## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.4356040599


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-6.3630953, -5.0424914, -6.3630953, -5.0424914, -0.7693038, 0.7693033)
1: (-14.1805782, -12.8190308, -14.1805782, -12.8190308, -0.9576340, 0.9576340)
2: (-7.3011007, -6.1059828, -7.3011007, -6.1059828, -0.8018274, 0.8018272)
3: (-3.6252294, -2.5504999, -3.6252294, -2.5504999, -0.8629718, 0.8629718)
4: (-8.9688091, -7.6955185, -8.9688091, -7.6955185, -0.9078436, 0.9078436)
5: (-4.2295380, -3.0515246, -4.2295380, -3.0515246, -0.6916952, 0.6916952)
6: (-4.7486496, -3.6884871, -4.7486496, -3.6884871, -0.7937036, 0.7937036)
7: (-12.0620222, -10.6761436, -12.0620222, -10.6761436, -0.9489627, 0.9489627)
8: (6.3367090, 7.5108032, 6.3367090, 7.5108032, -0.9051771, 0.9051766)
9: (-3.3238935, -2.4740212, -3.3238935, -2.4740212, -0.6065361, 0.6065361)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 24.62 + 33.77 = 58.38 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.4360401, upper bound: 0.4360405

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5804
type: DSZ, layer: 1, pos: 6142
type: DSZ, layer: 1, pos: 5814
type: DSZ, layer: 1, pos: 6210
type: DSZ, layer: 1, pos: 904
type: DSZ, layer: 1, pos: 514

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5804

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4353589, upper bound: 0.4360395
time: 3.84 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4360391, upper bound: 0.4353588
time: 4.59 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 8.44 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 8.44
Output dim: 8, lower bound: -0.4353589, upper bound: 0.4360395
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 8.44
Output dim: 8, lower bound: -0.4360391, upper bound: 0.4353588

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -6.3630953, -5.0424914, -6.3630953, -5.0424914, -0.7625649, 0.7642486
1: -14.1805782, -12.8190308, -14.1805782, -12.8190308, -0.9532204, 0.9544506
2: -7.3011007, -6.1059828, -7.3011007, -6.1059828, -0.7982955, 0.7966356
3: -3.6252294, -2.5504999, -3.6252294, -2.5504999, -0.8562346, 0.8585219
4: -8.9688091, -7.6955185, -8.9688091, -7.6955185, -0.9012270, 0.8986707
5: -4.2295380, -3.0515246, -4.2295380, -3.0515246, -0.6843076, 0.6861529
6: -4.7486496, -3.6884871, -4.7486496, -3.6884871, -0.7921515, 0.7927237
7: -12.0620222, -10.6761436, -12.0620222, -10.6761436, -0.9423785, 0.9397297
8: 6.3367090, 7.5108032, 6.3367090, 7.5108032, -0.9034166, 0.9042931
9: -3.3238935, -2.4740212, -3.3238935, -2.4740212, -0.6064429, 0.6062779

Time for backsubstitution: 23.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6210
type: DSZ, layer: 1, pos: 904
type: DSZ, layer: 1, pos: 6142
type: DSZ, layer: 1, pos: 5814
type: DSZ, layer: 1, pos: 514

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6210

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4343308, upper bound: 0.4360380
time: 3.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.4353575, upper bound: 0.4350110
time: 6.33 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -6.3630953, -5.0424914, -6.3630953, -5.0424914, -0.7642486, 0.7625649
1: -14.1805782, -12.8190308, -14.1805782, -12.8190308, -0.9544506, 0.9532204
2: -7.3011007, -6.1059828, -7.3011007, -6.1059828, -0.7966356, 0.7982960
3: -3.6252294, -2.5504999, -3.6252294, -2.5504999, -0.8585219, 0.8562346
4: -8.9688091, -7.6955185, -8.9688091, -7.6955185, -0.8986707, 0.9012270
5: -4.2295380, -3.0515246, -4.2295380, -3.0515246, -0.6861529, 0.6843078
6: -4.7486496, -3.6884871, -4.7486496, -3.6884871, -0.7927237, 0.7921515
7: -12.0620222, -10.6761436, -12.0620222, -10.6761436, -0.9397302, 0.9423785
8: 6.3367090, 7.5108032, 6.3367090, 7.5108032, -0.9042921, 0.9034176
9: -3.3238935, -2.4740212, -3.3238935, -2.4740212, -0.6062779, 0.6064429

Time for backsubstitution: 22.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6210
type: DSZ, layer: 1, pos: 6142
type: DSZ, layer: 1, pos: 5814
type: DSZ, layer: 1, pos: 904
type: DSZ, layer: 1, pos: 514

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6210

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.4350110, upper bound: 0.4353580
time: 3.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4360376, upper bound: 0.4343307
time: 5.17 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 31.40 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 31.40
Output dim: 8, lower bound: -0.4343308, upper bound: 0.4360380
DS_DSZ1_DSZ2, status: Status.VERIFIED, split count: 2, time: 31.40
Output dim: 8, lower bound: -0.4353575, upper bound: 0.4350110
DS_DSZ2_DSZ1, status: Status.VERIFIED, split count: 2, time: 31.40
Output dim: 8, lower bound: -0.4350110, upper bound: 0.4353580
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 31.40
Output dim: 8, lower bound: -0.4360376, upper bound: 0.4343307

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.3630953, -5.0424914, -6.3630953, -5.0424914, -0.7602835, 0.7572424
1: -14.1805782, -12.8190308, -14.1805782, -12.8190308, -0.9528723, 0.9533911
2: -7.3011007, -6.1059828, -7.3011007, -6.1059828, -0.7970352, 0.7927942
3: -3.6252294, -2.5504999, -3.6252294, -2.5504999, -0.8552308, 0.8581910
4: -8.9688091, -7.6955185, -8.9688091, -7.6955185, -0.8938298, 0.8962646
5: -4.2295380, -3.0515246, -4.2295380, -3.0515246, -0.6838610, 0.6847858
6: -4.7486496, -3.6884871, -4.7486496, -3.6884871, -0.7917042, 0.7925763
7: -12.0620222, -10.6761436, -12.0620222, -10.6761436, -0.9381094, 0.9383416
8: 6.3367090, 7.5108032, 6.3367090, 7.5108032, -0.9019785, 0.9038205
9: -3.3238935, -2.4740212, -3.3238935, -2.4740212, -0.6055911, 0.6036868

Time for backsubstitution: 22.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 904
type: DSZ, layer: 1, pos: 514
type: DSZ, layer: 1, pos: 5814
type: DSZ, layer: 1, pos: 6142

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 904

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4300994, upper bound: 0.4360370
time: 3.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.4343298, upper bound: 0.4318077
time: 5.07 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.3630953, -5.0424914, -6.3630953, -5.0424914, -0.7572422, 0.7602837
1: -14.1805782, -12.8190308, -14.1805782, -12.8190308, -0.9533911, 0.9528718
2: -7.3011007, -6.1059828, -7.3011007, -6.1059828, -0.7927942, 0.7970352
3: -3.6252294, -2.5504999, -3.6252294, -2.5504999, -0.8581910, 0.8552308
4: -8.9688091, -7.6955185, -8.9688091, -7.6955185, -0.8962646, 0.8938298
5: -4.2295380, -3.0515246, -4.2295380, -3.0515246, -0.6847861, 0.6838608
6: -4.7486496, -3.6884871, -4.7486496, -3.6884871, -0.7925763, 0.7917042
7: -12.0620222, -10.6761436, -12.0620222, -10.6761436, -0.9383416, 0.9381094
8: 6.3367090, 7.5108032, 6.3367090, 7.5108032, -0.9038210, 0.9019780
9: -3.3238935, -2.4740212, -3.3238935, -2.4740212, -0.6036868, 0.6055911

Time for backsubstitution: 22.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 904
type: DSZ, layer: 1, pos: 514
type: DSZ, layer: 1, pos: 6142
type: DSZ, layer: 1, pos: 5814

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 904

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.4318077, upper bound: 0.4343298
time: 5.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4360365, upper bound: 0.4300993
time: 4.53 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 32.52 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 32.52
Output dim: 8, lower bound: -0.4300994, upper bound: 0.4360370
DS_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 32.52
Output dim: 8, lower bound: -0.4343298, upper bound: 0.4318077
DS_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 32.52
Output dim: 8, lower bound: -0.4318077, upper bound: 0.4343298
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 32.52
Output dim: 8, lower bound: -0.4360365, upper bound: 0.4300993

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.3630953, -5.0424914, -6.3630953, -5.0424914, -0.7602227, 0.7571611
1: -14.1805782, -12.8190308, -14.1805782, -12.8190308, -0.9518237, 0.9520812
2: -7.3011007, -6.1059828, -7.3011007, -6.1059828, -0.7970347, 0.7927454
3: -3.6252294, -2.5504999, -3.6252294, -2.5504999, -0.8528743, 0.8564234
4: -8.9688091, -7.6955185, -8.9688091, -7.6955185, -0.8936582, 0.8960357
5: -4.2295380, -3.0515246, -4.2295380, -3.0515246, -0.6848834, 0.6861072
6: -4.7486496, -3.6884871, -4.7486496, -3.6884871, -0.7939754, 0.7955132
7: -12.0620222, -10.6761436, -12.0620222, -10.6761436, -0.9392133, 0.9391952
8: 6.3367090, 7.5108032, 6.3367090, 7.5108032, -0.9014921, 0.9034557
9: -3.3238935, -2.4740212, -3.3238935, -2.4740212, -0.6014524, 0.5981705

Time for backsubstitution: 22.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5814
type: DSZ, layer: 1, pos: 6142
type: DSZ, layer: 1, pos: 514

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5814

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4296450, upper bound: 0.4360358
time: 4.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.4300982, upper bound: 0.4355828
time: 3.87 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.3630953, -5.0424914, -6.3630953, -5.0424914, -0.7571609, 0.7602227
1: -14.1805782, -12.8190308, -14.1805782, -12.8190308, -0.9520812, 0.9518237
2: -7.3011007, -6.1059828, -7.3011007, -6.1059828, -0.7927456, 0.7970350
3: -3.6252294, -2.5504999, -3.6252294, -2.5504999, -0.8564234, 0.8528743
4: -8.9688091, -7.6955185, -8.9688091, -7.6955185, -0.8960357, 0.8936582
5: -4.2295380, -3.0515246, -4.2295380, -3.0515246, -0.6861074, 0.6848831
6: -4.7486496, -3.6884871, -4.7486496, -3.6884871, -0.7955132, 0.7939754
7: -12.0620222, -10.6761436, -12.0620222, -10.6761436, -0.9391952, 0.9392133
8: 6.3367090, 7.5108032, 6.3367090, 7.5108032, -0.9034557, 0.9014921
9: -3.3238935, -2.4740212, -3.3238935, -2.4740212, -0.5981705, 0.6014524

Time for backsubstitution: 22.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6142
type: DSZ, layer: 1, pos: 5814
type: DSZ, layer: 1, pos: 514

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6142

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.4350815, upper bound: 0.4300980
time: 3.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4360347, upper bound: 0.4291447
time: 3.50 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 30.08 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.08
Output dim: 8, lower bound: -0.4296450, upper bound: 0.4360358
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 30.08
Output dim: 8, lower bound: -0.4300982, upper bound: 0.4355828
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 30.08
Output dim: 8, lower bound: -0.4350815, upper bound: 0.4300980
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.08
Output dim: 8, lower bound: -0.4360347, upper bound: 0.4291447

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.3630953, -5.0424914, -6.3630953, -5.0424914, -0.7588282, 0.7526550
1: -14.1805782, -12.8190308, -14.1805782, -12.8190308, -0.9489808, 0.9526091
2: -7.3011007, -6.1059828, -7.3011007, -6.1059828, -0.7882423, 0.7810268
3: -3.6252294, -2.5504999, -3.6252294, -2.5504999, -0.8333006, 0.8417387
4: -8.9688091, -7.6955185, -8.9688091, -7.6955185, -0.8926220, 0.8946548
5: -4.2295380, -3.0515246, -4.2295380, -3.0515246, -0.6656122, 0.6719134
6: -4.7486496, -3.6884871, -4.7486496, -3.6884871, -0.7986507, 0.7977614
7: -12.0620222, -10.6761436, -12.0620222, -10.6761436, -0.9331632, 0.9311314
8: 6.3367090, 7.5108032, 6.3367090, 7.5108032, -0.9009628, 0.9035134
9: -3.3238935, -2.4740212, -3.3238935, -2.4740212, -0.5965321, 0.5912626

Time for backsubstitution: 23.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6142
type: DSZ, layer: 1, pos: 514

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6142

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4286901, upper bound: 0.4360334
time: 6.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.4296432, upper bound: 0.4350802
time: 5.98 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.3630953, -5.0424914, -6.3630953, -5.0424914, -0.7491164, 0.7542470
1: -14.1805782, -12.8190308, -14.1805782, -12.8190308, -0.9397173, 0.9425497
2: -7.3011007, -6.1059828, -7.3011007, -6.1059828, -0.7919688, 0.7958918
3: -3.6252294, -2.5504999, -3.6252294, -2.5504999, -0.8584127, 0.8556957
4: -8.9688091, -7.6955185, -8.9688091, -7.6955185, -0.8922448, 0.8908119
5: -4.2295380, -3.0515246, -4.2295380, -3.0515246, -0.6845207, 0.6822040
6: -4.7486496, -3.6884871, -4.7486496, -3.6884871, -0.7902856, 0.7870097
7: -12.0620222, -10.6761436, -12.0620222, -10.6761436, -0.9367747, 0.9373960
8: 6.3367090, 7.5108032, 6.3367090, 7.5108032, -0.9025459, 0.9002781
9: -3.3238935, -2.4740212, -3.3238935, -2.4740212, -0.5973232, 0.5991881

Time for backsubstitution: 23.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5814
type: DSZ, layer: 1, pos: 514

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5814

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.4355805, upper bound: 0.4291431
time: 6.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4360335, upper bound: 0.4286906
time: 3.99 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 34.13 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 34.13
Output dim: 8, lower bound: -0.4286901, upper bound: 0.4360334
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 34.13
Output dim: 8, lower bound: -0.4296432, upper bound: 0.4350802
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 34.13
Output dim: 8, lower bound: -0.4355805, upper bound: 0.4291431
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 34.13
Output dim: 8, lower bound: -0.4360335, upper bound: 0.4286906

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.3630953, -5.0424914, -6.3630953, -5.0424914, -0.7528529, 0.7446110
1: -14.1805782, -12.8190308, -14.1805782, -12.8190308, -0.9397063, 0.9402452
2: -7.3011007, -6.1059828, -7.3011007, -6.1059828, -0.7870979, 0.7802491
3: -3.6252294, -2.5504999, -3.6252294, -2.5504999, -0.8361225, 0.8437285
4: -8.9688091, -7.6955185, -8.9688091, -7.6955185, -0.8897753, 0.8908634
5: -4.2295380, -3.0515246, -4.2295380, -3.0515246, -0.6629329, 0.6703265
6: -4.7486496, -3.6884871, -4.7486496, -3.6884871, -0.7916856, 0.7925353
7: -12.0620222, -10.6761436, -12.0620222, -10.6761436, -0.9313459, 0.9287109
8: 6.3367090, 7.5108032, 6.3367090, 7.5108032, -0.8997478, 0.9026008
9: -3.3238935, -2.4740212, -3.3238935, -2.4740212, -0.5942683, 0.5904162

Time for backsubstitution: 23.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 514

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 514

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4254813, upper bound: 0.4360274
time: 6.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.4286843, upper bound: 0.4328243
time: 5.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.3630953, -5.0424914, -6.3630953, -5.0424914, -0.7446113, 0.7528527
1: -14.1805782, -12.8190308, -14.1805782, -12.8190308, -0.9402452, 0.9397063
2: -7.3011007, -6.1059828, -7.3011007, -6.1059828, -0.7802491, 0.7870979
3: -3.6252294, -2.5504999, -3.6252294, -2.5504999, -0.8437285, 0.8361225
4: -8.9688091, -7.6955185, -8.9688091, -7.6955185, -0.8908634, 0.8897753
5: -4.2295380, -3.0515246, -4.2295380, -3.0515246, -0.6703265, 0.6629329
6: -4.7486496, -3.6884871, -4.7486496, -3.6884871, -0.7925353, 0.7916856
7: -12.0620222, -10.6761436, -12.0620222, -10.6761436, -0.9287109, 0.9313459
8: 6.3367090, 7.5108032, 6.3367090, 7.5108032, -0.9026012, 0.8997469
9: -3.3238935, -2.4740212, -3.3238935, -2.4740212, -0.5904162, 0.5942683

Time for backsubstitution: 23.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 514

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 514

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.4328242, upper bound: 0.4286847
time: 3.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4360272, upper bound: 0.4254818
time: 3.91 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 31.05 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 31.05
Output dim: 8, lower bound: -0.4254813, upper bound: 0.4360274
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 31.05
Output dim: 8, lower bound: -0.4286843, upper bound: 0.4328243
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 31.05
Output dim: 8, lower bound: -0.4328242, upper bound: 0.4286847
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 31.05
Output dim: 8, lower bound: -0.4360272, upper bound: 0.4254818

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.3630953, -5.0424914, -6.3630953, -5.0424914, -0.7460802, 0.7395275
1: -14.1805782, -12.8190308, -14.1805782, -12.8190308, -0.9372749, 0.9362764
2: -7.3011007, -6.1059828, -7.3011007, -6.1059828, -0.7879171, 0.7818227
3: -3.6252294, -2.5504999, -3.6252294, -2.5504999, -0.8386283, 0.8450913
4: -8.9688091, -7.6955185, -8.9688091, -7.6955185, -0.8827834, 0.8815417
5: -4.2295380, -3.0515246, -4.2295380, -3.0515246, -0.6532900, 0.6574509
6: -4.7486496, -3.6884871, -4.7486496, -3.6884871, -0.7879038, 0.7860436
7: -12.0620222, -10.6761436, -12.0620222, -10.6761436, -0.9326468, 0.9305840
8: 6.3367090, 7.5108032, 6.3367090, 7.5108032, -0.8890409, 0.8953123
9: -3.3238935, -2.4740212, -3.3238935, -2.4740212, -0.5822761, 0.5819819

Time for backsubstitution: 23.38 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2227
type: DSZ, layer: 3, pos: 2615
type: DSZ, layer: 3, pos: 1849
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 697
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 2880
type: DSZ, layer: 3, pos: 164
type: DSZ, layer: 3, pos: 2144
type: DSZ, layer: 3, pos: 1976
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 418
type: DSZ, layer: 3, pos: 2824
type: DSZ, layer: 3, pos: 1702
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 1685
type: DSZ, layer: 3, pos: 600
type: DSZ, layer: 3, pos: 921
type: DSZ, layer: 3, pos: 911
type: DSZ, layer: 3, pos: 1970
type: DSZ, layer: 3, pos: 2519
type: DSZ, layer: 3, pos: 2878
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 1846

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2227

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3996926, upper bound: 0.4102416
time: 4.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3996926, upper bound: 0.4102416
time: 4.60 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.3630953, -5.0424914, -6.3630953, -5.0424914, -0.7395275, 0.7460804
1: -14.1805782, -12.8190308, -14.1805782, -12.8190308, -0.9362764, 0.9372749
2: -7.3011007, -6.1059828, -7.3011007, -6.1059828, -0.7818227, 0.7879167
3: -3.6252294, -2.5504999, -3.6252294, -2.5504999, -0.8450913, 0.8386283
4: -8.9688091, -7.6955185, -8.9688091, -7.6955185, -0.8815417, 0.8827834
5: -4.2295380, -3.0515246, -4.2295380, -3.0515246, -0.6574509, 0.6532900
6: -4.7486496, -3.6884871, -4.7486496, -3.6884871, -0.7860436, 0.7879033
7: -12.0620222, -10.6761436, -12.0620222, -10.6761436, -0.9305840, 0.9326468
8: 6.3367090, 7.5108032, 6.3367090, 7.5108032, -0.8953123, 0.8890409
9: -3.3238935, -2.4740212, -3.3238935, -2.4740212, -0.5819819, 0.5822761

Time for backsubstitution: 23.37 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 164
type: DSZ, layer: 3, pos: 600
type: DSZ, layer: 3, pos: 1970
type: DSZ, layer: 3, pos: 921
type: DSZ, layer: 3, pos: 2519
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 1685
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 2227
type: DSZ, layer: 3, pos: 697
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 2144
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 418
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 1976
type: DSZ, layer: 3, pos: 2878
type: DSZ, layer: 3, pos: 1849
type: DSZ, layer: 3, pos: 2880
type: DSZ, layer: 3, pos: 1846
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 2824
type: DSZ, layer: 3, pos: 1702
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 2615
type: DSZ, layer: 3, pos: 911

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 164

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.4151361, upper bound: 0.4199054
time: 3.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.4304466, upper bound: 0.4045971
time: 3.73 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 30.71 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 30.71
Output dim: 8, lower bound: -0.3996926, upper bound: 0.4102416
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 30.71
Output dim: 8, lower bound: -0.3996926, upper bound: 0.4102416
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 30.71
Output dim: 8, lower bound: -0.4151361, upper bound: 0.4199054
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 30.71
Output dim: 8, lower bound: -0.4304466, upper bound: 0.4045971

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 58.38 + 398.42 = 456.80 seconds
