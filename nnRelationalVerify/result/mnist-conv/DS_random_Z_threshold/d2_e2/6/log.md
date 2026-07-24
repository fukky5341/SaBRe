## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 6)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.23908329


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-6.2076020, -5.5223475, -6.2076020, -5.5223475, -0.3725266, 0.3725266)
1: (-6.1862726, -5.5502210, -6.1862726, -5.5502210, -0.3926246, 0.3926246)
2: (-9.9678802, -9.1544847, -9.9678802, -9.1544847, -0.3700438, 0.3700438)
3: (-8.3231497, -7.4329510, -8.3231497, -7.4329510, -0.5096865, 0.5096865)
4: (-7.5297031, -6.9073687, -7.5297031, -6.9073687, -0.3389231, 0.3389231)
5: (3.6488411, 4.4215889, 3.6488411, 4.4215889, -0.3426521, 0.3426521)
6: (-4.4604273, -3.7868159, -4.4604273, -3.7868159, -0.5508237, 0.5508237)
7: (-9.7019253, -9.0353975, -9.7019253, -9.0353975, -0.4555955, 0.4555957)
8: (0.4475598, 1.0476842, 0.4475598, 1.0476842, -0.4699328, 0.4699328)
9: (-4.8405838, -4.0761609, -4.8405838, -4.0761609, -0.5305023, 0.5305026)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 24.58 + 33.30 = 57.88 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.2656476, upper bound: 0.2656481

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 456
type: DSZ, layer: 1, pos: 6130
type: DSZ, layer: 1, pos: 51

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 456

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2656449, upper bound: 0.2632782
time: 2.95 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2632776, upper bound: 0.2656446
time: 2.88 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 5.84 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 5.84
Output dim: 5, lower bound: -0.2656449, upper bound: 0.2632782
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 5.84
Output dim: 5, lower bound: -0.2632776, upper bound: 0.2656446

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -6.2076020, -5.5223475, -6.2076020, -5.5223475, -0.3723462, 0.3724856
1: -6.1862726, -5.5502210, -6.1862726, -5.5502210, -0.3917334, 0.3924176
2: -9.9678802, -9.1544847, -9.9678802, -9.1544847, -0.3700407, 0.3700430
3: -8.3231497, -7.4329510, -8.3231497, -7.4329510, -0.5094280, 0.5085721
4: -7.5297031, -6.9073687, -7.5297031, -6.9073687, -0.3352710, 0.3380708
5: 3.6488411, 4.4215889, 3.6488411, 4.4215889, -0.3425300, 0.3421276
6: -4.4604273, -3.7868159, -4.4604273, -3.7868159, -0.5498054, 0.5505862
7: -9.7019253, -9.0353975, -9.7019253, -9.0353975, -0.4553814, 0.4546793
8: 0.4475598, 1.0476842, 0.4475598, 1.0476842, -0.4683299, 0.4695562
9: -4.8405838, -4.0761609, -4.8405838, -4.0761609, -0.5300360, 0.5284920

Time for backsubstitution: 22.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6130
type: DSZ, layer: 1, pos: 51

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6130

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2656437, upper bound: 0.2600264
time: 2.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2623166, upper bound: 0.2632764
time: 3.11 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -6.2076020, -5.5223475, -6.2076020, -5.5223475, -0.3724856, 0.3723460
1: -6.1862726, -5.5502210, -6.1862726, -5.5502210, -0.3924177, 0.3917335
2: -9.9678802, -9.1544847, -9.9678802, -9.1544847, -0.3700430, 0.3700407
3: -8.3231497, -7.4329510, -8.3231497, -7.4329510, -0.5085719, 0.5094280
4: -7.5297031, -6.9073687, -7.5297031, -6.9073687, -0.3380708, 0.3352710
5: 3.6488411, 4.4215889, 3.6488411, 4.4215889, -0.3421276, 0.3425300
6: -4.4604273, -3.7868159, -4.4604273, -3.7868159, -0.5505862, 0.5498054
7: -9.7019253, -9.0353975, -9.7019253, -9.0353975, -0.4546795, 0.4553812
8: 0.4475598, 1.0476842, 0.4475598, 1.0476842, -0.4695561, 0.4683298
9: -4.8405838, -4.0761609, -4.8405838, -4.0761609, -0.5284920, 0.5300357

Time for backsubstitution: 22.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 6130

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 51

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2629715, upper bound: 0.2608081
time: 2.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2629238, upper bound: 0.2641642
time: 2.93 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 28.41 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 28.41
Output dim: 5, lower bound: -0.2656437, upper bound: 0.2600264
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 28.41
Output dim: 5, lower bound: -0.2623166, upper bound: 0.2632764
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 28.41
Output dim: 5, lower bound: -0.2629715, upper bound: 0.2608081
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 28.41
Output dim: 5, lower bound: -0.2629238, upper bound: 0.2641642

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.2076020, -5.5223475, -6.2076020, -5.5223475, -0.3693974, 0.3717835
1: -6.1862726, -5.5502210, -6.1862726, -5.5502210, -0.3882654, 0.3915925
2: -9.9678802, -9.1544847, -9.9678802, -9.1544847, -0.3693581, 0.3698808
3: -8.3231497, -7.4329510, -8.3231497, -7.4329510, -0.5084200, 0.5083315
4: -7.5297031, -6.9073687, -7.5297031, -6.9073687, -0.3345895, 0.3379080
5: 3.6488411, 4.4215889, 3.6488411, 4.4215889, -0.3415669, 0.3380780
6: -4.4604273, -3.7868159, -4.4604273, -3.7868159, -0.5495629, 0.5495722
7: -9.7019253, -9.0353975, -9.7019253, -9.0353975, -0.4537821, 0.4542956
8: 0.4475598, 1.0476842, 0.4475598, 1.0476842, -0.4680939, 0.4685749
9: -4.8405838, -4.0761609, -4.8405838, -4.0761609, -0.5270834, 0.5277867

Time for backsubstitution: 23.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 51

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 51

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2641626, upper bound: 0.2596232
time: 2.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2608070, upper bound: 0.2596696
time: 3.07 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.2076020, -5.5223475, -6.2076020, -5.5223475, -0.3716438, 0.3695369
1: -6.1862726, -5.5502210, -6.1862726, -5.5502210, -0.3909082, 0.3889496
2: -9.9678802, -9.1544847, -9.9678802, -9.1544847, -0.3698784, 0.3693604
3: -8.3231497, -7.4329510, -8.3231497, -7.4329510, -0.5091877, 0.5075638
4: -7.5297031, -6.9073687, -7.5297031, -6.9073687, -0.3351083, 0.3373892
5: 3.6488411, 4.4215889, 3.6488411, 4.4215889, -0.3384805, 0.3411644
6: -4.4604273, -3.7868159, -4.4604273, -3.7868159, -0.5487914, 0.5503435
7: -9.7019253, -9.0353975, -9.7019253, -9.0353975, -0.4549975, 0.4530802
8: 0.4475598, 1.0476842, 0.4475598, 1.0476842, -0.4673486, 0.4693202
9: -4.8405838, -4.0761609, -4.8405838, -4.0761609, -0.5293303, 0.5255394

Time for backsubstitution: 23.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 51

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 51

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2608316, upper bound: 0.2629231
time: 2.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2575086, upper bound: 0.2629709
time: 3.18 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.2076020, -5.5223475, -6.2076020, -5.5223475, -0.3724716, 0.3723408
1: -6.1862726, -5.5502210, -6.1862726, -5.5502210, -0.3923886, 0.3917558
2: -9.9678802, -9.1544847, -9.9678802, -9.1544847, -0.3700351, 0.3700249
3: -8.3231497, -7.4329510, -8.3231497, -7.4329510, -0.5085816, 0.5094156
4: -7.5297031, -6.9073687, -7.5297031, -6.9073687, -0.3380263, 0.3353068
5: 3.6488411, 4.4215889, 3.6488411, 4.4215889, -0.3421488, 0.3425039
6: -4.4604273, -3.7868159, -4.4604273, -3.7868159, -0.5505445, 0.5497949
7: -9.7019253, -9.0353975, -9.7019253, -9.0353975, -0.4546890, 0.4553649
8: 0.4475598, 1.0476842, 0.4475598, 1.0476842, -0.4695358, 0.4683514
9: -4.8405838, -4.0761609, -4.8405838, -4.0761609, -0.5285206, 0.5299971

Time for backsubstitution: 22.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6130

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6130

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2629702, upper bound: 0.2575084
time: 2.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2596690, upper bound: 0.2608071
time: 3.06 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.2076020, -5.5223475, -6.2076020, -5.5223475, -0.3724804, 0.3723460
1: -6.1862726, -5.5502210, -6.1862726, -5.5502210, -0.3924177, 0.3917043
2: -9.9678802, -9.1544847, -9.9678802, -9.1544847, -0.3700272, 0.3700407
3: -8.3231497, -7.4329510, -8.3231497, -7.4329510, -0.5085595, 0.5094280
4: -7.5297031, -6.9073687, -7.5297031, -6.9073687, -0.3380708, 0.3352264
5: 3.6488411, 4.4215889, 3.6488411, 4.4215889, -0.3421015, 0.3425300
6: -4.4604273, -3.7868159, -4.4604273, -3.7868159, -0.5505757, 0.5498054
7: -9.7019253, -9.0353975, -9.7019253, -9.0353975, -0.4546633, 0.4553812
8: 0.4475598, 1.0476842, 0.4475598, 1.0476842, -0.4695561, 0.4683094
9: -4.8405838, -4.0761609, -4.8405838, -4.0761609, -0.5284534, 0.5300357

Time for backsubstitution: 22.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6130

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6130

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2629224, upper bound: 0.2608321
time: 3.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2596225, upper bound: 0.2641630
time: 3.09 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 29.06 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.06
Output dim: 5, lower bound: -0.2641626, upper bound: 0.2596232
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.06
Output dim: 5, lower bound: -0.2608070, upper bound: 0.2596696
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.06
Output dim: 5, lower bound: -0.2608316, upper bound: 0.2629231
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.06
Output dim: 5, lower bound: -0.2575086, upper bound: 0.2629709
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.06
Output dim: 5, lower bound: -0.2629702, upper bound: 0.2575084
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.06
Output dim: 5, lower bound: -0.2596690, upper bound: 0.2608071
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.06
Output dim: 5, lower bound: -0.2629224, upper bound: 0.2608321
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.06
Output dim: 5, lower bound: -0.2596225, upper bound: 0.2641630

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.2076020, -5.5223475, -6.2076020, -5.5223475, -0.3694012, 0.3717782
1: -6.1862726, -5.5502210, -6.1862726, -5.5502210, -0.3882365, 0.3916147
2: -9.9678802, -9.1544847, -9.9678802, -9.1544847, -0.3693547, 0.3698651
3: -8.3231497, -7.4329510, -8.3231497, -7.4329510, -0.5084295, 0.5083191
4: -7.5297031, -6.9073687, -7.5297031, -6.9073687, -0.3345453, 0.3379441
5: 3.6488411, 4.4215889, 3.6488411, 4.4215889, -0.3415883, 0.3380522
6: -4.4604273, -3.7868159, -4.4604273, -3.7868159, -0.5495832, 0.5495615
7: -9.7019253, -9.0353975, -9.7019253, -9.0353975, -0.4537940, 0.4542794
8: 0.4475598, 1.0476842, 0.4475598, 1.0476842, -0.4680734, 0.4685986
9: -4.8405838, -4.0761609, -4.8405838, -4.0761609, -0.5271120, 0.5277481

Time for backsubstitution: 22.78 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 598
type: DSZ, layer: 3, pos: 2144
type: DSZ, layer: 3, pos: 647
type: DSZ, layer: 3, pos: 2125
type: DSZ, layer: 3, pos: 1123
type: DSZ, layer: 3, pos: 698
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 2469
type: DSZ, layer: 3, pos: 2340
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 1733
type: DSZ, layer: 3, pos: 950
type: DSZ, layer: 3, pos: 1780
type: DSZ, layer: 3, pos: 1977
type: DSZ, layer: 3, pos: 312
type: DSZ, layer: 3, pos: 1457
type: DSZ, layer: 3, pos: 1679

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 417

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2633997, upper bound: 0.2590550
time: 2.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2636084, upper bound: 0.2588458
time: 2.84 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.2076020, -5.5223475, -6.2076020, -5.5223475, -0.3693922, 0.3717835
1: -6.1862726, -5.5502210, -6.1862726, -5.5502210, -0.3882654, 0.3915634
2: -9.9678802, -9.1544847, -9.9678802, -9.1544847, -0.3693423, 0.3698808
3: -8.3231497, -7.4329510, -8.3231497, -7.4329510, -0.5084076, 0.5083315
4: -7.5297031, -6.9073687, -7.5297031, -6.9073687, -0.3345895, 0.3378638
5: 3.6488411, 4.4215889, 3.6488411, 4.4215889, -0.3415408, 0.3380780
6: -4.4604273, -3.7868159, -4.4604273, -3.7868159, -0.5495522, 0.5495722
7: -9.7019253, -9.0353975, -9.7019253, -9.0353975, -0.4537659, 0.4542956
8: 0.4475598, 1.0476842, 0.4475598, 1.0476842, -0.4680939, 0.4685543
9: -4.8405838, -4.0761609, -4.8405838, -4.0761609, -0.5270443, 0.5277867

Time for backsubstitution: 23.23 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 950
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 698
type: DSZ, layer: 3, pos: 2144
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 2340
type: DSZ, layer: 3, pos: 1679
type: DSZ, layer: 3, pos: 1457
type: DSZ, layer: 3, pos: 2469
type: DSZ, layer: 3, pos: 647
type: DSZ, layer: 3, pos: 1977
type: DSZ, layer: 3, pos: 1123
type: DSZ, layer: 3, pos: 2125
type: DSZ, layer: 3, pos: 598
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 1733
type: DSZ, layer: 3, pos: 1780
type: DSZ, layer: 3, pos: 312
type: DSZ, layer: 3, pos: 2130

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 950

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2608033, upper bound: 0.2369706
time: 3.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2342644, upper bound: 0.2596647
time: 3.43 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.2076020, -5.5223475, -6.2076020, -5.5223475, -0.3716476, 0.3695316
1: -6.1862726, -5.5502210, -6.1862726, -5.5502210, -0.3908792, 0.3889720
2: -9.9678802, -9.1544847, -9.9678802, -9.1544847, -0.3698751, 0.3693446
3: -8.3231497, -7.4329510, -8.3231497, -7.4329510, -0.5091972, 0.5075514
4: -7.5297031, -6.9073687, -7.5297031, -6.9073687, -0.3350641, 0.3374254
5: 3.6488411, 4.4215889, 3.6488411, 4.4215889, -0.3385018, 0.3411385
6: -4.4604273, -3.7868159, -4.4604273, -3.7868159, -0.5488117, 0.5503330
7: -9.7019253, -9.0353975, -9.7019253, -9.0353975, -0.4550099, 0.4530640
8: 0.4475598, 1.0476842, 0.4475598, 1.0476842, -0.4673281, 0.4693439
9: -4.8405838, -4.0761609, -4.8405838, -4.0761609, -0.5293593, 0.5255008

Time for backsubstitution: 23.51 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 312
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 2125
type: DSZ, layer: 3, pos: 698
type: DSZ, layer: 3, pos: 598
type: DSZ, layer: 3, pos: 2144
type: DSZ, layer: 3, pos: 1679
type: DSZ, layer: 3, pos: 1977
type: DSZ, layer: 3, pos: 647
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 1733
type: DSZ, layer: 3, pos: 950
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 1457
type: DSZ, layer: 3, pos: 1780
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 2469
type: DSZ, layer: 3, pos: 1123
type: DSZ, layer: 3, pos: 2340

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 312

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2287995, upper bound: 0.2277364
time: 2.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2259028, upper bound: 0.2306291
time: 2.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.2076020, -5.5223475, -6.2076020, -5.5223475, -0.3716388, 0.3695369
1: -6.1862726, -5.5502210, -6.1862726, -5.5502210, -0.3909082, 0.3889208
2: -9.9678802, -9.1544847, -9.9678802, -9.1544847, -0.3698627, 0.3693604
3: -8.3231497, -7.4329510, -8.3231497, -7.4329510, -0.5091753, 0.5075638
4: -7.5297031, -6.9073687, -7.5297031, -6.9073687, -0.3351083, 0.3373450
5: 3.6488411, 4.4215889, 3.6488411, 4.4215889, -0.3384546, 0.3411644
6: -4.4604273, -3.7868159, -4.4604273, -3.7868159, -0.5487807, 0.5503435
7: -9.7019253, -9.0353975, -9.7019253, -9.0353975, -0.4549813, 0.4530802
8: 0.4475598, 1.0476842, 0.4475598, 1.0476842, -0.4673486, 0.4692996
9: -4.8405838, -4.0761609, -4.8405838, -4.0761609, -0.5292921, 0.5255394

Time for backsubstitution: 23.26 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 1123
type: DSZ, layer: 3, pos: 1733
type: DSZ, layer: 3, pos: 1977
type: DSZ, layer: 3, pos: 1679
type: DSZ, layer: 3, pos: 1780
type: DSZ, layer: 3, pos: 312
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 598
type: DSZ, layer: 3, pos: 2340
type: DSZ, layer: 3, pos: 1457
type: DSZ, layer: 3, pos: 2144
type: DSZ, layer: 3, pos: 2469
type: DSZ, layer: 3, pos: 647
type: DSZ, layer: 3, pos: 950
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 698
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 2125

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2130

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2541086, upper bound: 0.2584896
time: 3.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2530302, upper bound: 0.2595692
time: 2.82 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.2076020, -5.5223475, -6.2076020, -5.5223475, -0.3695228, 0.3716387
1: -6.1862726, -5.5502210, -6.1862726, -5.5502210, -0.3889205, 0.3909307
2: -9.9678802, -9.1544847, -9.9678802, -9.1544847, -0.3693525, 0.3698627
3: -8.3231497, -7.4329510, -8.3231497, -7.4329510, -0.5075736, 0.5091751
4: -7.5297031, -6.9073687, -7.5297031, -6.9073687, -0.3373450, 0.3351444
5: 3.6488411, 4.4215889, 3.6488411, 4.4215889, -0.3411858, 0.3384545
6: -4.4604273, -3.7868159, -4.4604273, -3.7868159, -0.5503018, 0.5487807
7: -9.7019253, -9.0353975, -9.7019253, -9.0353975, -0.4530902, 0.4549813
8: 0.4475598, 1.0476842, 0.4475598, 1.0476842, -0.4692996, 0.4673698
9: -4.8405838, -4.0761609, -4.8405838, -4.0761609, -0.5255680, 0.5292919

Time for backsubstitution: 23.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1977
type: DSZ, layer: 3, pos: 647
type: DSZ, layer: 3, pos: 2469
type: DSZ, layer: 3, pos: 1457
type: DSZ, layer: 3, pos: 1733
type: DSZ, layer: 3, pos: 1123
type: DSZ, layer: 3, pos: 1679
type: DSZ, layer: 3, pos: 950
type: DSZ, layer: 3, pos: 1780
type: DSZ, layer: 3, pos: 312
type: DSZ, layer: 3, pos: 2125
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 2144
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 698
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 598
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 2340

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1977

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2626739, upper bound: 0.2561102
time: 3.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2614675, upper bound: 0.2572123
time: 2.96 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.2076020, -5.5223475, -6.2076020, -5.5223475, -0.3717692, 0.3693920
1: -6.1862726, -5.5502210, -6.1862726, -5.5502210, -0.3915634, 0.3882878
2: -9.9678802, -9.1544847, -9.9678802, -9.1544847, -0.3698728, 0.3693423
3: -8.3231497, -7.4329510, -8.3231497, -7.4329510, -0.5083413, 0.5084074
4: -7.5297031, -6.9073687, -7.5297031, -6.9073687, -0.3378638, 0.3346256
5: 3.6488411, 4.4215889, 3.6488411, 4.4215889, -0.3380994, 0.3415409
6: -4.4604273, -3.7868159, -4.4604273, -3.7868159, -0.5495303, 0.5495522
7: -9.7019253, -9.0353975, -9.7019253, -9.0353975, -0.4543056, 0.4537659
8: 0.4475598, 1.0476842, 0.4475598, 1.0476842, -0.4685543, 0.4681151
9: -4.8405838, -4.0761609, -4.8405838, -4.0761609, -0.5278158, 0.5270445

Time for backsubstitution: 23.22 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1977
type: DSZ, layer: 3, pos: 2125
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 2144
type: DSZ, layer: 3, pos: 312
type: DSZ, layer: 3, pos: 1457
type: DSZ, layer: 3, pos: 647
type: DSZ, layer: 3, pos: 2469
type: DSZ, layer: 3, pos: 1733
type: DSZ, layer: 3, pos: 1780
type: DSZ, layer: 3, pos: 1679
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 950
type: DSZ, layer: 3, pos: 1123
type: DSZ, layer: 3, pos: 598
type: DSZ, layer: 3, pos: 2340
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 698

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1977

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2593716, upper bound: 0.2594115
time: 2.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2581653, upper bound: 0.2605123
time: 3.14 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.2076020, -5.5223475, -6.2076020, -5.5223475, -0.3695316, 0.3716439
1: -6.1862726, -5.5502210, -6.1862726, -5.5502210, -0.3889496, 0.3908794
2: -9.9678802, -9.1544847, -9.9678802, -9.1544847, -0.3693446, 0.3698785
3: -8.3231497, -7.4329510, -8.3231497, -7.4329510, -0.5075514, 0.5091877
4: -7.5297031, -6.9073687, -7.5297031, -6.9073687, -0.3373892, 0.3350641
5: 3.6488411, 4.4215889, 3.6488411, 4.4215889, -0.3411384, 0.3384805
6: -4.4604273, -3.7868159, -4.4604273, -3.7868159, -0.5503330, 0.5487914
7: -9.7019253, -9.0353975, -9.7019253, -9.0353975, -0.4530640, 0.4549975
8: 0.4475598, 1.0476842, 0.4475598, 1.0476842, -0.4693201, 0.4673278
9: -4.8405838, -4.0761609, -4.8405838, -4.0761609, -0.5255008, 0.5293305

Time for backsubstitution: 23.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 647
type: DSZ, layer: 3, pos: 1123
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 1679
type: DSZ, layer: 3, pos: 2125
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 1780
type: DSZ, layer: 3, pos: 1457
type: DSZ, layer: 3, pos: 2340
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 1733
type: DSZ, layer: 3, pos: 1977
type: DSZ, layer: 3, pos: 312
type: DSZ, layer: 3, pos: 2469
type: DSZ, layer: 3, pos: 598
type: DSZ, layer: 3, pos: 698
type: DSZ, layer: 3, pos: 950
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 2144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 647

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2622284, upper bound: 0.2533210
time: 3.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2540555, upper bound: 0.2601136
time: 2.99 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.2076020, -5.5223475, -6.2076020, -5.5223475, -0.3717782, 0.3693972
1: -6.1862726, -5.5502210, -6.1862726, -5.5502210, -0.3915923, 0.3882365
2: -9.9678802, -9.1544847, -9.9678802, -9.1544847, -0.3698651, 0.3693581
3: -8.3231497, -7.4329510, -8.3231497, -7.4329510, -0.5083189, 0.5084200
4: -7.5297031, -6.9073687, -7.5297031, -6.9073687, -0.3379080, 0.3345453
5: 3.6488411, 4.4215889, 3.6488411, 4.4215889, -0.3380522, 0.3415668
6: -4.4604273, -3.7868159, -4.4604273, -3.7868159, -0.5495615, 0.5495629
7: -9.7019253, -9.0353975, -9.7019253, -9.0353975, -0.4542794, 0.4537821
8: 0.4475598, 1.0476842, 0.4475598, 1.0476842, -0.4685748, 0.4680731
9: -4.8405838, -4.0761609, -4.8405838, -4.0761609, -0.5277481, 0.5270832

Time for backsubstitution: 23.17 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 647
type: DSZ, layer: 3, pos: 1679
type: DSZ, layer: 3, pos: 312
type: DSZ, layer: 3, pos: 950
type: DSZ, layer: 3, pos: 2469
type: DSZ, layer: 3, pos: 598
type: DSZ, layer: 3, pos: 1123
type: DSZ, layer: 3, pos: 698
type: DSZ, layer: 3, pos: 2340
type: DSZ, layer: 3, pos: 2125
type: DSZ, layer: 3, pos: 1977
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 1780
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 1733
type: DSZ, layer: 3, pos: 2144
type: DSZ, layer: 3, pos: 1457
type: DSZ, layer: 3, pos: 2866

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 647

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2589907, upper bound: 0.2566582
time: 2.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2507407, upper bound: 0.2633583
time: 3.33 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 29.46 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.46
Output dim: 5, lower bound: -0.2633997, upper bound: 0.2590550
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.46
Output dim: 5, lower bound: -0.2636084, upper bound: 0.2588458
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.46
Output dim: 5, lower bound: -0.2608033, upper bound: 0.2369706
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.46
Output dim: 5, lower bound: -0.2342644, upper bound: 0.2596647
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 29.46
Output dim: 5, lower bound: -0.2287995, upper bound: 0.2277364
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 29.46
Output dim: 5, lower bound: -0.2259028, upper bound: 0.2306291
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.46
Output dim: 5, lower bound: -0.2541086, upper bound: 0.2584896
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.46
Output dim: 5, lower bound: -0.2530302, upper bound: 0.2595692
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.46
Output dim: 5, lower bound: -0.2626739, upper bound: 0.2561102
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.46
Output dim: 5, lower bound: -0.2614675, upper bound: 0.2572123
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.46
Output dim: 5, lower bound: -0.2593716, upper bound: 0.2594115
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.46
Output dim: 5, lower bound: -0.2581653, upper bound: 0.2605123
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.46
Output dim: 5, lower bound: -0.2622284, upper bound: 0.2533210
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.46
Output dim: 5, lower bound: -0.2540555, upper bound: 0.2601136
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.46
Output dim: 5, lower bound: -0.2589907, upper bound: 0.2566582
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.46
Output dim: 5, lower bound: -0.2507407, upper bound: 0.2633583

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.2076020, -5.5223475, -6.2076020, -5.5223475, -0.3693502, 0.3717992
1: -6.1862726, -5.5502210, -6.1862726, -5.5502210, -0.3881886, 0.3916179
2: -9.9678802, -9.1544847, -9.9678802, -9.1544847, -0.3692933, 0.3698382
3: -8.3231497, -7.4329510, -8.3231497, -7.4329510, -0.5083332, 0.5082998
4: -7.5297031, -6.9073687, -7.5297031, -6.9073687, -0.3345187, 0.3379179
5: 3.6488411, 4.4215889, 3.6488411, 4.4215889, -0.3414743, 0.3380090
6: -4.4604273, -3.7868159, -4.4604273, -3.7868159, -0.5495548, 0.5496140
7: -9.7019253, -9.0353975, -9.7019253, -9.0353975, -0.4537771, 0.4542885
8: 0.4475598, 1.0476842, 0.4475598, 1.0476842, -0.4676616, 0.4683778
9: -4.8405838, -4.0761609, -4.8405838, -4.0761609, -0.5271001, 0.5277412

Time for backsubstitution: 22.85 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1679
type: DSZ, layer: 3, pos: 2144
type: DSZ, layer: 3, pos: 1457
type: DSZ, layer: 3, pos: 312
type: DSZ, layer: 3, pos: 2340
type: DSZ, layer: 3, pos: 598
type: DSZ, layer: 3, pos: 1123
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 950
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 2125
type: DSZ, layer: 3, pos: 1780
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 1733
type: DSZ, layer: 3, pos: 2469
type: DSZ, layer: 3, pos: 1977
type: DSZ, layer: 3, pos: 647
type: DSZ, layer: 3, pos: 698

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1679

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2590460, upper bound: 0.2576128
time: 2.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2590460, upper bound: 0.2576128
time: 2.98 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.2076020, -5.5223475, -6.2076020, -5.5223475, -0.3694012, 0.3717275
1: -6.1862726, -5.5502210, -6.1862726, -5.5502210, -0.3882365, 0.3915669
2: -9.9678802, -9.1544847, -9.9678802, -9.1544847, -0.3693279, 0.3698651
3: -8.3231497, -7.4329510, -8.3231497, -7.4329510, -0.5084295, 0.5082226
4: -7.5297031, -6.9073687, -7.5297031, -6.9073687, -0.3345189, 0.3379441
5: 3.6488411, 4.4215889, 3.6488411, 4.4215889, -0.3415883, 0.3379382
6: -4.4604273, -3.7868159, -4.4604273, -3.7868159, -0.5495832, 0.5495329
7: -9.7019253, -9.0353975, -9.7019253, -9.0353975, -0.4537940, 0.4542625
8: 0.4475598, 1.0476842, 0.4475598, 1.0476842, -0.4678524, 0.4685986
9: -4.8405838, -4.0761609, -4.8405838, -4.0761609, -0.5271053, 0.5277481

Time for backsubstitution: 22.47 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1733
type: DSZ, layer: 3, pos: 647
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 950
type: DSZ, layer: 3, pos: 2340
type: DSZ, layer: 3, pos: 1679
type: DSZ, layer: 3, pos: 2125
type: DSZ, layer: 3, pos: 1457
type: DSZ, layer: 3, pos: 698
type: DSZ, layer: 3, pos: 1123
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 2144
type: DSZ, layer: 3, pos: 312
type: DSZ, layer: 3, pos: 1977
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 1780
type: DSZ, layer: 3, pos: 598
type: DSZ, layer: 3, pos: 2469

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1733

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2388555, upper bound: 0.2355657
time: 2.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2388555, upper bound: 0.2355657
time: 2.73 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.2076020, -5.5223475, -6.2076020, -5.5223475, -0.3693652, 0.3717597
1: -6.1862726, -5.5502210, -6.1862726, -5.5502210, -0.3882668, 0.3915666
2: -9.9678802, -9.1544847, -9.9678802, -9.1544847, -0.3693441, 0.3698853
3: -8.3231497, -7.4329510, -8.3231497, -7.4329510, -0.5083983, 0.5083272
4: -7.5297031, -6.9073687, -7.5297031, -6.9073687, -0.3345863, 0.3378625
5: 3.6488411, 4.4215889, 3.6488411, 4.4215889, -0.3415313, 0.3380609
6: -4.4604273, -3.7868159, -4.4604273, -3.7868159, -0.5495481, 0.5495670
7: -9.7019253, -9.0353975, -9.7019253, -9.0353975, -0.4537368, 0.4542587
8: 0.4475598, 1.0476842, 0.4475598, 1.0476842, -0.4680860, 0.4685500
9: -4.8405838, -4.0761609, -4.8405838, -4.0761609, -0.5270529, 0.5278018

Time for backsubstitution: 22.58 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1457
type: DSZ, layer: 3, pos: 2469
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 698
type: DSZ, layer: 3, pos: 2125
type: DSZ, layer: 3, pos: 2340
type: DSZ, layer: 3, pos: 312
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 1780
type: DSZ, layer: 3, pos: 2144
type: DSZ, layer: 3, pos: 1733
type: DSZ, layer: 3, pos: 647
type: DSZ, layer: 3, pos: 1123
type: DSZ, layer: 3, pos: 1679
type: DSZ, layer: 3, pos: 598
type: DSZ, layer: 3, pos: 1977

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1457

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2579563, upper bound: 0.2363836
time: 3.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2602097, upper bound: 0.2341304
time: 3.39 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.2076020, -5.5223475, -6.2076020, -5.5223475, -0.3693683, 0.3717539
1: -6.1862726, -5.5502210, -6.1862726, -5.5502210, -0.3882682, 0.3915645
2: -9.9678802, -9.1544847, -9.9678802, -9.1544847, -0.3693466, 0.3698825
3: -8.3231497, -7.4329510, -8.3231497, -7.4329510, -0.5084033, 0.5083213
4: -7.5297031, -6.9073687, -7.5297031, -6.9073687, -0.3345875, 0.3378605
5: 3.6488411, 4.4215889, 3.6488411, 4.4215889, -0.3415222, 0.3380687
6: -4.4604273, -3.7868159, -4.4604273, -3.7868159, -0.5495470, 0.5495677
7: -9.7019253, -9.0353975, -9.7019253, -9.0353975, -0.4537287, 0.4542649
8: 0.4475598, 1.0476842, 0.4475598, 1.0476842, -0.4680896, 0.4685454
9: -4.8405838, -4.0761609, -4.8405838, -4.0761609, -0.5270567, 0.5277948

Time for backsubstitution: 22.61 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1123
type: DSZ, layer: 3, pos: 2469
type: DSZ, layer: 3, pos: 698
type: DSZ, layer: 3, pos: 647
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 1457
type: DSZ, layer: 3, pos: 2144
type: DSZ, layer: 3, pos: 598
type: DSZ, layer: 3, pos: 1977
type: DSZ, layer: 3, pos: 1780
type: DSZ, layer: 3, pos: 2125
type: DSZ, layer: 3, pos: 312
type: DSZ, layer: 3, pos: 1679
type: DSZ, layer: 3, pos: 2340
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 1733

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1123

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2339929, upper bound: 0.2575610
time: 3.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2321644, upper bound: 0.2593961
time: 3.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.2076020, -5.5223475, -6.2076020, -5.5223475, -0.3728771, 0.3680996
1: -6.1862726, -5.5502210, -6.1862726, -5.5502210, -0.3918567, 0.3896711
2: -9.9678802, -9.1544847, -9.9678802, -9.1544847, -0.3749794, 0.3708651
3: -8.3231497, -7.4329510, -8.3231497, -7.4329510, -0.4906249, 0.4884155
4: -7.5297031, -6.9073687, -7.5297031, -6.9073687, -0.3323213, 0.3345296
5: 3.6488411, 4.4215889, 3.6488411, 4.4215889, -0.3112736, 0.3138824
6: -4.4604273, -3.7868159, -4.4604273, -3.7868159, -0.5467744, 0.5479970
7: -9.7019253, -9.0353975, -9.7019253, -9.0353975, -0.4549017, 0.4530017
8: 0.4475598, 1.0476842, 0.4475598, 1.0476842, -0.4691815, 0.4703519
9: -4.8405838, -4.0761609, -4.8405838, -4.0761609, -0.5243549, 0.5190835

Time for backsubstitution: 25.17 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 57.88 + 553.02 = 610.90 seconds
