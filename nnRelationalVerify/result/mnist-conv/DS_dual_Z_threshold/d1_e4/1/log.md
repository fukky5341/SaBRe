## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.3552878565


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-7.3612347, -6.1413960, -7.3612347, -6.1413960, -0.7229214, 0.7229214)
1: (-6.7012358, -5.3436651, -6.7012358, -5.3436651, -1.1029615, 1.1029606)
2: (-4.7904668, -3.6836765, -4.7904668, -3.6836765, -0.8028684, 0.8028684)
3: (-5.1534758, -3.8338423, -5.1534758, -3.8338423, -0.7839131, 0.7839131)
4: (-10.7349405, -9.5796747, -10.7349405, -9.5796747, -0.6515818, 0.6515818)
5: (1.3470602, 2.2432051, 1.3470602, 2.2432051, -0.6103714, 0.6103711)
6: (0.1716712, 1.3331558, 0.1716712, 1.3331558, -0.6167908, 0.6167908)
7: (-12.6086416, -11.2634993, -12.6086416, -11.2634993, -0.8833489, 0.8833489)
8: (6.0859089, 7.0352283, 6.0859089, 7.0352283, -0.8516474, 0.8516474)
9: (-8.6910915, -7.4509592, -8.6910915, -7.4509592, -1.0465469, 1.0465469)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.18 + 35.05 = 57.22 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.3556433, upper bound: 0.3556435

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6182
type: DSZ, layer: 1, pos: 4614
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 6212
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 536

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 6182

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3550295, upper bound: 0.3556418
time: 6.40 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3556426, upper bound: 0.3550296
time: 5.61 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 12.24 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 12.24
Output dim: 8, lower bound: -0.3550295, upper bound: 0.3556418
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 12.24
Output dim: 8, lower bound: -0.3556426, upper bound: 0.3550296

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -7.3612347, -6.1413960, -7.3612347, -6.1413960, -0.7228956, 0.7229023
1: -6.7012358, -5.3436651, -6.7012358, -5.3436651, -1.0989981, 1.1000433
2: -4.7904668, -3.6836765, -4.7904668, -3.6836765, -0.8026209, 0.8025341
3: -5.1534758, -3.8338423, -5.1534758, -3.8338423, -0.7805614, 0.7814417
4: -10.7349405, -9.5796747, -10.7349405, -9.5796747, -0.6492038, 0.6483555
5: 1.3470602, 2.2432051, 1.3470602, 2.2432051, -0.6103060, 0.6102829
6: 0.1716712, 1.3331558, 0.1716712, 1.3331558, -0.6150093, 0.6143713
7: -12.6086416, -11.2634993, -12.6086416, -11.2634993, -0.8823962, 0.8826470
8: 6.0859089, 7.0352283, 6.0859089, 7.0352283, -0.8499832, 0.8504205
9: -8.6910915, -7.4509592, -8.6910915, -7.4509592, -1.0442853, 1.0434785

Time for backsubstitution: 21.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4614
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 6212
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 536

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 4614

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3550292, upper bound: 0.3556266
time: 4.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3550192, upper bound: 0.3556424
time: 4.48 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -7.3612347, -6.1413960, -7.3612347, -6.1413960, -0.7229023, 0.7228951
1: -6.7012358, -5.3436651, -6.7012358, -5.3436651, -1.1000433, 1.0989981
2: -4.7904668, -3.6836765, -4.7904668, -3.6836765, -0.8025341, 0.8026209
3: -5.1534758, -3.8338423, -5.1534758, -3.8338423, -0.7814417, 0.7805614
4: -10.7349405, -9.5796747, -10.7349405, -9.5796747, -0.6483550, 0.6492043
5: 1.3470602, 2.2432051, 1.3470602, 2.2432051, -0.6102827, 0.6103060
6: 0.1716712, 1.3331558, 0.1716712, 1.3331558, -0.6143713, 0.6150093
7: -12.6086416, -11.2634993, -12.6086416, -11.2634993, -0.8826470, 0.8823962
8: 6.0859089, 7.0352283, 6.0859089, 7.0352283, -0.8504205, 0.8499827
9: -8.6910915, -7.4509592, -8.6910915, -7.4509592, -1.0434785, 1.0442848

Time for backsubstitution: 21.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4614
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 6212
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 536

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 4614

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3556421, upper bound: 0.3550187
time: 7.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3556259, upper bound: 0.3550299
time: 7.32 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 35.93 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 35.93
Output dim: 8, lower bound: -0.3550292, upper bound: 0.3556266
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 35.93
Output dim: 8, lower bound: -0.3550192, upper bound: 0.3556424
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 35.93
Output dim: 8, lower bound: -0.3556421, upper bound: 0.3550187
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 35.93
Output dim: 8, lower bound: -0.3556259, upper bound: 0.3550299

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.3612347, -6.1413960, -7.3612347, -6.1413960, -0.7216897, 0.7215271
1: -6.7012358, -5.3436651, -6.7012358, -5.3436651, -1.0889864, 1.0918217
2: -4.7904668, -3.6836765, -4.7904668, -3.6836765, -0.7858877, 0.7878923
3: -5.1534758, -3.8338423, -5.1534758, -3.8338423, -0.7738953, 0.7728519
4: -10.7349405, -9.5796747, -10.7349405, -9.5796747, -0.6486263, 0.6476965
5: 1.3470602, 2.2432051, 1.3470602, 2.2432051, -0.6116536, 0.6111615
6: 0.1716712, 1.3331558, 0.1716712, 1.3331558, -0.6109564, 0.6094587
7: -12.6086416, -11.2634993, -12.6086416, -11.2634993, -0.8806338, 0.8818231
8: 6.0859089, 7.0352283, 6.0859089, 7.0352283, -0.8522639, 0.8523374
9: -8.6910915, -7.4509592, -8.6910915, -7.4509592, -1.0401912, 1.0387993

Time for backsubstitution: 21.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 6212
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 536

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 522

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3537414, upper bound: 0.3556220
time: 5.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3550252, upper bound: 0.3543315
time: 4.91 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.3612347, -6.1413960, -7.3612347, -6.1413960, -0.7215199, 0.7216964
1: -6.7012358, -5.3436651, -6.7012358, -5.3436651, -1.0907793, 1.0900326
2: -4.7904668, -3.6836765, -4.7904668, -3.6836765, -0.7879801, 0.7858009
3: -5.1534758, -3.8338423, -5.1534758, -3.8338423, -0.7719717, 0.7747765
4: -10.7349405, -9.5796747, -10.7349405, -9.5796747, -0.6485453, 0.6477771
5: 1.3470602, 2.2432051, 1.3470602, 2.2432051, -0.6111844, 0.6116312
6: 0.1716712, 1.3331558, 0.1716712, 1.3331558, -0.6100972, 0.6103194
7: -12.6086416, -11.2634993, -12.6086416, -11.2634993, -0.8815732, 0.8808851
8: 6.0859089, 7.0352283, 6.0859089, 7.0352283, -0.8519001, 0.8527017
9: -8.6910915, -7.4509592, -8.6910915, -7.4509592, -1.0396061, 1.0393858

Time for backsubstitution: 21.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 6212
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 536

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 522

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3537261, upper bound: 0.3556376
time: 9.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3550152, upper bound: 0.3543474
time: 6.90 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.3612347, -6.1413960, -7.3612347, -6.1413960, -0.7216964, 0.7215199
1: -6.7012358, -5.3436651, -6.7012358, -5.3436651, -1.0900316, 1.0907803
2: -4.7904668, -3.6836765, -4.7904668, -3.6836765, -0.7858009, 0.7879801
3: -5.1534758, -3.8338423, -5.1534758, -3.8338423, -0.7747765, 0.7719717
4: -10.7349405, -9.5796747, -10.7349405, -9.5796747, -0.6477776, 0.6485453
5: 1.3470602, 2.2432051, 1.3470602, 2.2432051, -0.6116312, 0.6111846
6: 0.1716712, 1.3331558, 0.1716712, 1.3331558, -0.6103194, 0.6100967
7: -12.6086416, -11.2634993, -12.6086416, -11.2634993, -0.8808846, 0.8815737
8: 6.0859089, 7.0352283, 6.0859089, 7.0352283, -0.8527017, 0.8518996
9: -8.6910915, -7.4509592, -8.6910915, -7.4509592, -1.0393858, 1.0396061

Time for backsubstitution: 21.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 6212
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 536

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 522

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3543474, upper bound: 0.3550145
time: 5.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3556381, upper bound: 0.3537268
time: 4.59 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.3612347, -6.1413960, -7.3612347, -6.1413960, -0.7215266, 0.7216897
1: -6.7012358, -5.3436651, -6.7012358, -5.3436651, -1.0918226, 1.0889874
2: -4.7904668, -3.6836765, -4.7904668, -3.6836765, -0.7878923, 0.7858877
3: -5.1534758, -3.8338423, -5.1534758, -3.8338423, -0.7728519, 0.7738953
4: -10.7349405, -9.5796747, -10.7349405, -9.5796747, -0.6476965, 0.6486263
5: 1.3470602, 2.2432051, 1.3470602, 2.2432051, -0.6111615, 0.6116538
6: 0.1716712, 1.3331558, 0.1716712, 1.3331558, -0.6094592, 0.6109562
7: -12.6086416, -11.2634993, -12.6086416, -11.2634993, -0.8818231, 0.8806343
8: 6.0859089, 7.0352283, 6.0859089, 7.0352283, -0.8523374, 0.8522639
9: -8.6910915, -7.4509592, -8.6910915, -7.4509592, -1.0387993, 1.0401912

Time for backsubstitution: 22.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 6212
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 536

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 522

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3543319, upper bound: 0.3550255
time: 4.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3556222, upper bound: 0.3537422
time: 4.74 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 31.76 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.76
Output dim: 8, lower bound: -0.3537414, upper bound: 0.3556220
DS_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 31.76
Output dim: 8, lower bound: -0.3550252, upper bound: 0.3543315
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.76
Output dim: 8, lower bound: -0.3537261, upper bound: 0.3556376
DS_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 31.76
Output dim: 8, lower bound: -0.3550152, upper bound: 0.3543474
DS_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 31.76
Output dim: 8, lower bound: -0.3543474, upper bound: 0.3550145
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.76
Output dim: 8, lower bound: -0.3556381, upper bound: 0.3537268
DS_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 31.76
Output dim: 8, lower bound: -0.3543319, upper bound: 0.3550255
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.76
Output dim: 8, lower bound: -0.3556222, upper bound: 0.3537422

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.3612347, -6.1413960, -7.3612347, -6.1413960, -0.7218418, 0.7210245
1: -6.7012358, -5.3436651, -6.7012358, -5.3436651, -1.0893135, 1.0907421
2: -4.7904668, -3.6836765, -4.7904668, -3.6836765, -0.7865963, 0.7855468
3: -5.1534758, -3.8338423, -5.1534758, -3.8338423, -0.7692113, 0.7742677
4: -10.7349405, -9.5796747, -10.7349405, -9.5796747, -0.6483088, 0.6477914
5: 1.3470602, 2.2432051, 1.3470602, 2.2432051, -0.6118774, 0.6104128
6: 0.1716712, 1.3331558, 0.1716712, 1.3331558, -0.6119905, 0.6060357
7: -12.6086416, -11.2634993, -12.6086416, -11.2634993, -0.8806596, 0.8817387
8: 6.0859089, 7.0352283, 6.0859089, 7.0352283, -0.8508415, 0.8527679
9: -8.6910915, -7.4509592, -8.6910915, -7.4509592, -1.0403872, 1.0381484

Time for backsubstitution: 21.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6212
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 536

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 6212

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3532046, upper bound: 0.3556206
time: 5.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3537402, upper bound: 0.3550873
time: 5.13 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.3612347, -6.1413960, -7.3612347, -6.1413960, -0.7216725, 0.7211943
1: -6.7012358, -5.3436651, -6.7012358, -5.3436651, -1.0911064, 1.0889530
2: -4.7904668, -3.6836765, -4.7904668, -3.6836765, -0.7886891, 0.7834554
3: -5.1534758, -3.8338423, -5.1534758, -3.8338423, -0.7672877, 0.7761922
4: -10.7349405, -9.5796747, -10.7349405, -9.5796747, -0.6482277, 0.6478724
5: 1.3470602, 2.2432051, 1.3470602, 2.2432051, -0.6114087, 0.6108825
6: 0.1716712, 1.3331558, 0.1716712, 1.3331558, -0.6111307, 0.6068964
7: -12.6086416, -11.2634993, -12.6086416, -11.2634993, -0.8815989, 0.8808002
8: 6.0859089, 7.0352283, 6.0859089, 7.0352283, -0.8504777, 0.8531318
9: -8.6910915, -7.4509592, -8.6910915, -7.4509592, -1.0398016, 1.0387344

Time for backsubstitution: 21.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6212
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 536

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 6212

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3531887, upper bound: 0.3556362
time: 5.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3537248, upper bound: 0.3551035
time: 6.50 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.3612347, -6.1413960, -7.3612347, -6.1413960, -0.7211943, 0.7215199
1: -6.7012358, -5.3436651, -6.7012358, -5.3436651, -1.0889530, 1.0907803
2: -4.7904668, -3.6836765, -4.7904668, -3.6836765, -0.7834558, 0.7879801
3: -5.1534758, -3.8338423, -5.1534758, -3.8338423, -0.7747765, 0.7672875
4: -10.7349405, -9.5796747, -10.7349405, -9.5796747, -0.6477776, 0.6482282
5: 1.3470602, 2.2432051, 1.3470602, 2.2432051, -0.6108828, 0.6111846
6: 0.1716712, 1.3331558, 0.1716712, 1.3331558, -0.6068964, 0.6100967
7: -12.6086416, -11.2634993, -12.6086416, -11.2634993, -0.8808002, 0.8815737
8: 6.0859089, 7.0352283, 6.0859089, 7.0352283, -0.8527017, 0.8504777
9: -8.6910915, -7.4509592, -8.6910915, -7.4509592, -1.0387344, 1.0396061

Time for backsubstitution: 21.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6212
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 536

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 6212

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3551028, upper bound: 0.3537247
time: 4.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3556368, upper bound: 0.3531894
time: 4.38 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.3612347, -6.1413960, -7.3612347, -6.1413960, -0.7210245, 0.7216897
1: -6.7012358, -5.3436651, -6.7012358, -5.3436651, -1.0907421, 1.0889874
2: -4.7904668, -3.6836765, -4.7904668, -3.6836765, -0.7855468, 0.7858877
3: -5.1534758, -3.8338423, -5.1534758, -3.8338423, -0.7728519, 0.7692111
4: -10.7349405, -9.5796747, -10.7349405, -9.5796747, -0.6476965, 0.6483092
5: 1.3470602, 2.2432051, 1.3470602, 2.2432051, -0.6104126, 0.6116538
6: 0.1716712, 1.3331558, 0.1716712, 1.3331558, -0.6060357, 0.6109562
7: -12.6086416, -11.2634993, -12.6086416, -11.2634993, -0.8817387, 0.8806343
8: 6.0859089, 7.0352283, 6.0859089, 7.0352283, -0.8523374, 0.8508415
9: -8.6910915, -7.4509592, -8.6910915, -7.4509592, -1.0381484, 1.0401912

Time for backsubstitution: 21.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6212
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 536

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 6212

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3550876, upper bound: 0.3537401
time: 4.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3556209, upper bound: 0.3532053
time: 4.55 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 30.77 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.77
Output dim: 8, lower bound: -0.3532046, upper bound: 0.3556206
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 30.77
Output dim: 8, lower bound: -0.3537402, upper bound: 0.3550873
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.77
Output dim: 8, lower bound: -0.3531887, upper bound: 0.3556362
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 30.77
Output dim: 8, lower bound: -0.3537248, upper bound: 0.3551035
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 30.77
Output dim: 8, lower bound: -0.3551028, upper bound: 0.3537247
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.77
Output dim: 8, lower bound: -0.3556368, upper bound: 0.3531894
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 30.77
Output dim: 8, lower bound: -0.3550876, upper bound: 0.3537401
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.77
Output dim: 8, lower bound: -0.3556209, upper bound: 0.3532053

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.3612347, -6.1413960, -7.3612347, -6.1413960, -0.7180033, 0.7182293
1: -6.7012358, -5.3436651, -6.7012358, -5.3436651, -1.0888481, 1.0901041
2: -4.7904668, -3.6836765, -4.7904668, -3.6836765, -0.7828398, 0.7803941
3: -5.1534758, -3.8338423, -5.1534758, -3.8338423, -0.7689872, 0.7741032
4: -10.7349405, -9.5796747, -10.7349405, -9.5796747, -0.6445818, 0.6450777
5: 1.3470602, 2.2432051, 1.3470602, 2.2432051, -0.6096172, 0.6087654
6: 0.1716712, 1.3331558, 0.1716712, 1.3331558, -0.6097448, 0.6029506
7: -12.6086416, -11.2634993, -12.6086416, -11.2634993, -0.8789735, 0.8805108
8: 6.0859089, 7.0352283, 6.0859089, 7.0352283, -0.8491755, 0.8515525
9: -8.6910915, -7.4509592, -8.6910915, -7.4509592, -1.0388012, 1.0369935

Time for backsubstitution: 21.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 536

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 542

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3532022, upper bound: 0.3541883
time: 4.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3517744, upper bound: 0.3556191
time: 4.48 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.3612347, -6.1413960, -7.3612347, -6.1413960, -0.7178340, 0.7183990
1: -6.7012358, -5.3436651, -6.7012358, -5.3436651, -1.0906410, 1.0883131
2: -4.7904668, -3.6836765, -4.7904668, -3.6836765, -0.7849321, 0.7783031
3: -5.1534758, -3.8338423, -5.1534758, -3.8338423, -0.7670636, 0.7760277
4: -10.7349405, -9.5796747, -10.7349405, -9.5796747, -0.6445007, 0.6451588
5: 1.3470602, 2.2432051, 1.3470602, 2.2432051, -0.6091480, 0.6092350
6: 0.1716712, 1.3331558, 0.1716712, 1.3331558, -0.6088850, 0.6038113
7: -12.6086416, -11.2634993, -12.6086416, -11.2634993, -0.8799129, 0.8795724
8: 6.0859089, 7.0352283, 6.0859089, 7.0352283, -0.8488111, 0.8519163
9: -8.6910915, -7.4509592, -8.6910915, -7.4509592, -1.0382161, 1.0375800

Time for backsubstitution: 20.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 536

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 542

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3531863, upper bound: 0.3542024
time: 7.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3517591, upper bound: 0.3556350
time: 4.62 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.3612347, -6.1413960, -7.3612347, -6.1413960, -0.7183990, 0.7176814
1: -6.7012358, -5.3436651, -6.7012358, -5.3436651, -1.0883131, 1.0903130
2: -4.7904668, -3.6836765, -4.7904668, -3.6836765, -0.7783031, 0.7842236
3: -5.1534758, -3.8338423, -5.1534758, -3.8338423, -0.7746129, 0.7670636
4: -10.7349405, -9.5796747, -10.7349405, -9.5796747, -0.6450634, 0.6445007
5: 1.3470602, 2.2432051, 1.3470602, 2.2432051, -0.6092348, 0.6089244
6: 0.1716712, 1.3331558, 0.1716712, 1.3331558, -0.6038115, 0.6078513
7: -12.6086416, -11.2634993, -12.6086416, -11.2634993, -0.8795724, 0.8798871
8: 6.0859089, 7.0352283, 6.0859089, 7.0352283, -0.8514862, 0.8488111
9: -8.6910915, -7.4509592, -8.6910915, -7.4509592, -1.0375800, 1.0380201

Time for backsubstitution: 20.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 536

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 542

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3556348, upper bound: 0.3517598
time: 4.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3542027, upper bound: 0.3531870
time: 4.66 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.3612347, -6.1413960, -7.3612347, -6.1413960, -0.7182293, 0.7178507
1: -6.7012358, -5.3436651, -6.7012358, -5.3436651, -1.0901041, 1.0885201
2: -4.7904668, -3.6836765, -4.7904668, -3.6836765, -0.7803941, 0.7821312
3: -5.1534758, -3.8338423, -5.1534758, -3.8338423, -0.7726879, 0.7689872
4: -10.7349405, -9.5796747, -10.7349405, -9.5796747, -0.6449823, 0.6445818
5: 1.3470602, 2.2432051, 1.3470602, 2.2432051, -0.6087656, 0.6093934
6: 0.1716712, 1.3331558, 0.1716712, 1.3331558, -0.6029508, 0.6087108
7: -12.6086416, -11.2634993, -12.6086416, -11.2634993, -0.8805108, 0.8789477
8: 6.0859089, 7.0352283, 6.0859089, 7.0352283, -0.8511219, 0.8491755
9: -8.6910915, -7.4509592, -8.6910915, -7.4509592, -1.0369935, 1.0386052

Time for backsubstitution: 21.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 536

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 542

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3556189, upper bound: 0.3517751
time: 4.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3541885, upper bound: 0.3532027
time: 4.40 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 30.48 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 30.48
Output dim: 8, lower bound: -0.3532022, upper bound: 0.3541883
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 30.48
Output dim: 8, lower bound: -0.3517744, upper bound: 0.3556191
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 30.48
Output dim: 8, lower bound: -0.3531863, upper bound: 0.3542024
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 30.48
Output dim: 8, lower bound: -0.3517591, upper bound: 0.3556350
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 30.48
Output dim: 8, lower bound: -0.3556348, upper bound: 0.3517598
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 30.48
Output dim: 8, lower bound: -0.3542027, upper bound: 0.3531870
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 30.48
Output dim: 8, lower bound: -0.3556189, upper bound: 0.3517751
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 30.48
Output dim: 8, lower bound: -0.3541885, upper bound: 0.3532027

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.3612347, -6.1413960, -7.3612347, -6.1413960, -0.7036057, 0.7017798
1: -6.7012358, -5.3436651, -6.7012358, -5.3436651, -1.0828495, 1.0832491
2: -4.7904668, -3.6836765, -4.7904668, -3.6836765, -0.7744851, 0.7706165
3: -5.1534758, -3.8338423, -5.1534758, -3.8338423, -0.7456608, 0.7536912
4: -10.7349405, -9.5796747, -10.7349405, -9.5796747, -0.6149263, 0.6111603
5: 1.3470602, 2.2432051, 1.3470602, 2.2432051, -0.5916586, 0.5930500
6: 0.1716712, 1.3331558, 0.1716712, 1.3331558, -0.5925326, 0.5879779
7: -12.6086416, -11.2634993, -12.6086416, -11.2634993, -0.8644862, 0.8639579
8: 6.0859089, 7.0352283, 6.0859089, 7.0352283, -0.8389778, 0.8426285
9: -8.6910915, -7.4509592, -8.6910915, -7.4509592, -1.0336585, 1.0311170

Time for backsubstitution: 20.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 536

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 536

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3514120, upper bound: 0.3556164
time: 5.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3517731, upper bound: 0.3552600
time: 5.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.3612347, -6.1413960, -7.3612347, -6.1413960, -0.7034359, 0.7019496
1: -6.7012358, -5.3436651, -6.7012358, -5.3436651, -1.0846424, 1.0814600
2: -4.7904668, -3.6836765, -4.7904668, -3.6836765, -0.7765775, 0.7685256
3: -5.1534758, -3.8338423, -5.1534758, -3.8338423, -0.7437372, 0.7556157
4: -10.7349405, -9.5796747, -10.7349405, -9.5796747, -0.6148453, 0.6112411
5: 1.3470602, 2.2432051, 1.3470602, 2.2432051, -0.5911899, 0.5935197
6: 0.1716712, 1.3331558, 0.1716712, 1.3331558, -0.5916734, 0.5888386
7: -12.6086416, -11.2634993, -12.6086416, -11.2634993, -0.8654256, 0.8630195
8: 6.0859089, 7.0352283, 6.0859089, 7.0352283, -0.8386140, 0.8429928
9: -8.6910915, -7.4509592, -8.6910915, -7.4509592, -1.0330734, 1.0317030

Time for backsubstitution: 20.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 536

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 536

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3513954, upper bound: 0.3556314
time: 4.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3517579, upper bound: 0.3552747
time: 5.02 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.3612347, -6.1413960, -7.3612347, -6.1413960, -0.7019496, 0.7032833
1: -6.7012358, -5.3436651, -6.7012358, -5.3436651, -1.0814590, 1.0843153
2: -4.7904668, -3.6836765, -4.7904668, -3.6836765, -0.7685256, 0.7758679
3: -5.1534758, -3.8338423, -5.1534758, -3.8338423, -0.7542000, 0.7437367
4: -10.7349405, -9.5796747, -10.7349405, -9.5796747, -0.6111465, 0.6148453
5: 1.3470602, 2.2432051, 1.3470602, 2.2432051, -0.5935197, 0.5909655
6: 0.1716712, 1.3331558, 0.1716712, 1.3331558, -0.5888381, 0.5906398
7: -12.6086416, -11.2634993, -12.6086416, -11.2634993, -0.8630195, 0.8653998
8: 6.0859089, 7.0352283, 6.0859089, 7.0352283, -0.8425627, 0.8386135
9: -8.6910915, -7.4509592, -8.6910915, -7.4509592, -1.0317030, 1.0328765

Time for backsubstitution: 20.87 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 57.22 + 550.96 = 608.19 seconds
