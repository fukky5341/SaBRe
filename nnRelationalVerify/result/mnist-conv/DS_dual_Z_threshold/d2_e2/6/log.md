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
execution time: IAR + RelationalAnalysis = 23.13 + 33.40 = 56.53 seconds
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

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 456

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2656449, upper bound: 0.2632782
time: 3.04 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2632776, upper bound: 0.2656446
time: 2.97 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 6.23 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 6.23
Output dim: 5, lower bound: -0.2656449, upper bound: 0.2632782
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 6.23
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

Time for backsubstitution: 20.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6130
type: DSZ, layer: 1, pos: 51

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 6130

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2656437, upper bound: 0.2600264
time: 2.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2623166, upper bound: 0.2632764
time: 3.07 seconds

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

Time for backsubstitution: 20.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6130
type: DSZ, layer: 1, pos: 51

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 6130

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2632765, upper bound: 0.2623164
time: 3.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2600258, upper bound: 0.2656436
time: 2.84 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 27.04 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 27.04
Output dim: 5, lower bound: -0.2656437, upper bound: 0.2600264
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 27.04
Output dim: 5, lower bound: -0.2623166, upper bound: 0.2632764
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 27.04
Output dim: 5, lower bound: -0.2632765, upper bound: 0.2623164
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 27.04
Output dim: 5, lower bound: -0.2600258, upper bound: 0.2656436

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

Time for backsubstitution: 22.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 51

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 51

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2641626, upper bound: 0.2596232
time: 3.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2608070, upper bound: 0.2596696
time: 3.39 seconds

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

Time for backsubstitution: 22.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 51

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 51

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2608316, upper bound: 0.2629231
time: 3.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2575086, upper bound: 0.2629709
time: 3.35 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.2076020, -5.5223475, -6.2076020, -5.5223475, -0.3695369, 0.3716439
1: -6.1862726, -5.5502210, -6.1862726, -5.5502210, -0.3889496, 0.3909082
2: -9.9678802, -9.1544847, -9.9678802, -9.1544847, -0.3693604, 0.3698785
3: -8.3231497, -7.4329510, -8.3231497, -7.4329510, -0.5075638, 0.5091877
4: -7.5297031, -6.9073687, -7.5297031, -6.9073687, -0.3373892, 0.3351083
5: 3.6488411, 4.4215889, 3.6488411, 4.4215889, -0.3411645, 0.3384805
6: -4.4604273, -3.7868159, -4.4604273, -3.7868159, -0.5503438, 0.5487914
7: -9.7019253, -9.0353975, -9.7019253, -9.0353975, -0.4530802, 0.4549975
8: 0.4475598, 1.0476842, 0.4475598, 1.0476842, -0.4693201, 0.4673485
9: -4.8405838, -4.0761609, -4.8405838, -4.0761609, -0.5255394, 0.5293305

Time for backsubstitution: 21.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 51

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 51

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2629702, upper bound: 0.2575084
time: 3.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2629224, upper bound: 0.2608321
time: 3.17 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.2076020, -5.5223475, -6.2076020, -5.5223475, -0.3717835, 0.3693972
1: -6.1862726, -5.5502210, -6.1862726, -5.5502210, -0.3915923, 0.3882656
2: -9.9678802, -9.1544847, -9.9678802, -9.1544847, -0.3698808, 0.3693581
3: -8.3231497, -7.4329510, -8.3231497, -7.4329510, -0.5083315, 0.5084200
4: -7.5297031, -6.9073687, -7.5297031, -6.9073687, -0.3379080, 0.3345895
5: 3.6488411, 4.4215889, 3.6488411, 4.4215889, -0.3380781, 0.3415668
6: -4.4604273, -3.7868159, -4.4604273, -3.7868159, -0.5495722, 0.5495629
7: -9.7019253, -9.0353975, -9.7019253, -9.0353975, -0.4542956, 0.4537821
8: 0.4475598, 1.0476842, 0.4475598, 1.0476842, -0.4685748, 0.4680938
9: -4.8405838, -4.0761609, -4.8405838, -4.0761609, -0.5277867, 0.5270832

Time for backsubstitution: 21.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 51

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 51

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2596690, upper bound: 0.2608071
time: 3.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2596225, upper bound: 0.2641630
time: 3.37 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 28.28 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.28
Output dim: 5, lower bound: -0.2641626, upper bound: 0.2596232
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.28
Output dim: 5, lower bound: -0.2608070, upper bound: 0.2596696
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.28
Output dim: 5, lower bound: -0.2608316, upper bound: 0.2629231
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.28
Output dim: 5, lower bound: -0.2575086, upper bound: 0.2629709
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.28
Output dim: 5, lower bound: -0.2629702, upper bound: 0.2575084
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.28
Output dim: 5, lower bound: -0.2629224, upper bound: 0.2608321
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.28
Output dim: 5, lower bound: -0.2596690, upper bound: 0.2608071
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.28
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

Time for backsubstitution: 21.75 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2125
type: DSZ, layer: 3, pos: 1977
type: DSZ, layer: 3, pos: 2144
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 312
type: DSZ, layer: 3, pos: 698
type: DSZ, layer: 3, pos: 2340
type: DSZ, layer: 3, pos: 1733
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 1123
type: DSZ, layer: 3, pos: 1780
type: DSZ, layer: 3, pos: 598
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 647
type: DSZ, layer: 3, pos: 1679
type: DSZ, layer: 3, pos: 1457
type: DSZ, layer: 3, pos: 2469
type: DSZ, layer: 3, pos: 950

Time for candidate selection: 0.38 seconds

### Candidate
type: DSZ, layer: 3, pos: 2125

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2463982, upper bound: 0.2481345
time: 3.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2537190, upper bound: 0.2452723
time: 3.32 seconds

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

Time for backsubstitution: 22.34 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2125
type: DSZ, layer: 3, pos: 1977
type: DSZ, layer: 3, pos: 2144
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 312
type: DSZ, layer: 3, pos: 698
type: DSZ, layer: 3, pos: 2340
type: DSZ, layer: 3, pos: 1733
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 1123
type: DSZ, layer: 3, pos: 1780
type: DSZ, layer: 3, pos: 598
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 647
type: DSZ, layer: 3, pos: 1679
type: DSZ, layer: 3, pos: 1457
type: DSZ, layer: 3, pos: 2469
type: DSZ, layer: 3, pos: 950

Time for candidate selection: 0.48 seconds

### Candidate
type: DSZ, layer: 3, pos: 2125

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2444188, upper bound: 0.2481967
time: 3.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2494894, upper bound: 0.2432635
time: 3.50 seconds

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

Time for backsubstitution: 22.40 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2125
type: DSZ, layer: 3, pos: 1977
type: DSZ, layer: 3, pos: 2144
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 312
type: DSZ, layer: 3, pos: 698
type: DSZ, layer: 3, pos: 2340
type: DSZ, layer: 3, pos: 1733
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 1123
type: DSZ, layer: 3, pos: 1780
type: DSZ, layer: 3, pos: 598
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 647
type: DSZ, layer: 3, pos: 1679
type: DSZ, layer: 3, pos: 1457
type: DSZ, layer: 3, pos: 2469
type: DSZ, layer: 3, pos: 950

Time for candidate selection: 0.50 seconds

### Candidate
type: DSZ, layer: 3, pos: 2125

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2429514, upper bound: 0.2515125
time: 4.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2502880, upper bound: 0.2486011
time: 3.12 seconds

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

Time for backsubstitution: 22.14 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2125
type: DSZ, layer: 3, pos: 1977
type: DSZ, layer: 3, pos: 2144
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 312
type: DSZ, layer: 3, pos: 698
type: DSZ, layer: 3, pos: 2340
type: DSZ, layer: 3, pos: 1733
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 1123
type: DSZ, layer: 3, pos: 1780
type: DSZ, layer: 3, pos: 598
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 647
type: DSZ, layer: 3, pos: 1679
type: DSZ, layer: 3, pos: 1457
type: DSZ, layer: 3, pos: 2469
type: DSZ, layer: 3, pos: 950

Time for candidate selection: 0.38 seconds

### Candidate
type: DSZ, layer: 3, pos: 2125

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2410319, upper bound: 0.2515762
time: 3.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2461115, upper bound: 0.2466805
time: 2.86 seconds

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

Time for backsubstitution: 21.82 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2125
type: DSZ, layer: 3, pos: 1977
type: DSZ, layer: 3, pos: 2144
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 312
type: DSZ, layer: 3, pos: 698
type: DSZ, layer: 3, pos: 2340
type: DSZ, layer: 3, pos: 1733
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 1123
type: DSZ, layer: 3, pos: 1780
type: DSZ, layer: 3, pos: 598
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 647
type: DSZ, layer: 3, pos: 1679
type: DSZ, layer: 3, pos: 1457
type: DSZ, layer: 3, pos: 2469
type: DSZ, layer: 3, pos: 950

Time for candidate selection: 0.34 seconds

### Candidate
type: DSZ, layer: 3, pos: 2125

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 1977

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2626739, upper bound: 0.2561102
time: 3.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2614675, upper bound: 0.2572123
time: 3.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 21.65 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2125
type: DSZ, layer: 3, pos: 1977
type: DSZ, layer: 3, pos: 2144
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 312
type: DSZ, layer: 3, pos: 698
type: DSZ, layer: 3, pos: 2340
type: DSZ, layer: 3, pos: 1733
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 1123
type: DSZ, layer: 3, pos: 1780
type: DSZ, layer: 3, pos: 598
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 647
type: DSZ, layer: 3, pos: 1679
type: DSZ, layer: 3, pos: 1457
type: DSZ, layer: 3, pos: 2469
type: DSZ, layer: 3, pos: 950

Time for candidate selection: 0.35 seconds

### Candidate
type: DSZ, layer: 3, pos: 2125

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2486009, upper bound: 0.2502884
time: 3.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2515119, upper bound: 0.2429519
time: 3.52 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 21.88 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2125
type: DSZ, layer: 3, pos: 1977
type: DSZ, layer: 3, pos: 2144
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 312
type: DSZ, layer: 3, pos: 698
type: DSZ, layer: 3, pos: 2340
type: DSZ, layer: 3, pos: 1733
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 1123
type: DSZ, layer: 3, pos: 1780
type: DSZ, layer: 3, pos: 598
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 647
type: DSZ, layer: 3, pos: 1679
type: DSZ, layer: 3, pos: 1457
type: DSZ, layer: 3, pos: 2469
type: DSZ, layer: 3, pos: 950

Time for candidate selection: 0.46 seconds

### Candidate
type: DSZ, layer: 3, pos: 2125

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2432631, upper bound: 0.2494900
time: 3.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2481963, upper bound: 0.2444186
time: 3.04 seconds

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

Time for backsubstitution: 21.74 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2125
type: DSZ, layer: 3, pos: 1977
type: DSZ, layer: 3, pos: 2144
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 312
type: DSZ, layer: 3, pos: 698
type: DSZ, layer: 3, pos: 2340
type: DSZ, layer: 3, pos: 1733
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 1123
type: DSZ, layer: 3, pos: 1780
type: DSZ, layer: 3, pos: 598
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 647
type: DSZ, layer: 3, pos: 1679
type: DSZ, layer: 3, pos: 1457
type: DSZ, layer: 3, pos: 2469
type: DSZ, layer: 3, pos: 950

Time for candidate selection: 0.35 seconds

### Candidate
type: DSZ, layer: 3, pos: 2125

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2452717, upper bound: 0.2537189
time: 3.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2481342, upper bound: 0.2463988
time: 3.05 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 28.45 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.45
Output dim: 5, lower bound: -0.2463982, upper bound: 0.2481345
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.45
Output dim: 5, lower bound: -0.2537190, upper bound: 0.2452723
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.45
Output dim: 5, lower bound: -0.2444188, upper bound: 0.2481967
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.45
Output dim: 5, lower bound: -0.2494894, upper bound: 0.2432635
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.45
Output dim: 5, lower bound: -0.2429514, upper bound: 0.2515125
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.45
Output dim: 5, lower bound: -0.2502880, upper bound: 0.2486011
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.45
Output dim: 5, lower bound: -0.2410319, upper bound: 0.2515762
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.45
Output dim: 5, lower bound: -0.2461115, upper bound: 0.2466805
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.45
Output dim: 5, lower bound: -0.2626739, upper bound: 0.2561102
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.45
Output dim: 5, lower bound: -0.2614675, upper bound: 0.2572123
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.45
Output dim: 5, lower bound: -0.2486009, upper bound: 0.2502884
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.45
Output dim: 5, lower bound: -0.2515119, upper bound: 0.2429519
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.45
Output dim: 5, lower bound: -0.2432631, upper bound: 0.2494900
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.45
Output dim: 5, lower bound: -0.2481963, upper bound: 0.2444186
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.45
Output dim: 5, lower bound: -0.2452717, upper bound: 0.2537189
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.45
Output dim: 5, lower bound: -0.2481342, upper bound: 0.2463988

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.2076020, -5.5223475, -6.2076020, -5.5223475, -0.3585289, 0.3623633
1: -6.1862726, -5.5502210, -6.1862726, -5.5502210, -0.3850911, 0.3870516
2: -9.9678802, -9.1544847, -9.9678802, -9.1544847, -0.3597224, 0.3577911
3: -8.3231497, -7.4329510, -8.3231497, -7.4329510, -0.4710791, 0.4610131
4: -7.5297031, -6.9073687, -7.5297031, -6.9073687, -0.3337318, 0.3357474
5: 3.6488411, 4.4215889, 3.6488411, 4.4215889, -0.3175107, 0.3172879
6: -4.4604273, -3.7868159, -4.4604273, -3.7868159, -0.5366406, 0.5416887
7: -9.7019253, -9.0353975, -9.7019253, -9.0353975, -0.3934159, 0.4025402
8: 0.4475598, 1.0476842, 0.4475598, 1.0476842, -0.4475992, 0.4493964
9: -4.8405838, -4.0761609, -4.8405838, -4.0761609, -0.4938564, 0.5028689

Time for backsubstitution: 21.69 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1977
type: DSZ, layer: 3, pos: 2144
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 312
type: DSZ, layer: 3, pos: 698
type: DSZ, layer: 3, pos: 2340
type: DSZ, layer: 3, pos: 1733
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 1123
type: DSZ, layer: 3, pos: 1780
type: DSZ, layer: 3, pos: 598
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 647
type: DSZ, layer: 3, pos: 1679
type: DSZ, layer: 3, pos: 1457
type: DSZ, layer: 3, pos: 2469
type: DSZ, layer: 3, pos: 950

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 1977

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2460605, upper bound: 0.2471210
time: 3.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2452909, upper bound: 0.2478575
time: 3.44 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.2076020, -5.5223475, -6.2076020, -5.5223475, -0.3599863, 0.3630226
1: -6.1862726, -5.5502210, -6.1862726, -5.5502210, -0.3843215, 0.3884695
2: -9.9678802, -9.1544847, -9.9678802, -9.1544847, -0.3589745, 0.3602327
3: -8.3231497, -7.4329510, -8.3231497, -7.4329510, -0.4611237, 0.4791546
4: -7.5297031, -6.9073687, -7.5297031, -6.9073687, -0.3328995, 0.3371307
5: 3.6488411, 4.4215889, 3.6488411, 4.4215889, -0.3248043, 0.3139747
6: -4.4604273, -3.7868159, -4.4604273, -3.7868159, -0.5417104, 0.5385504
7: -9.7019253, -9.0353975, -9.7019253, -9.0353975, -0.4020548, 0.4009471
8: 0.4475598, 1.0476842, 0.4475598, 1.0476842, -0.4488714, 0.4512820
9: -4.8405838, -4.0761609, -4.8405838, -4.0761609, -0.5022326, 0.5001252

Time for backsubstitution: 21.74 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1977
type: DSZ, layer: 3, pos: 2144
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 312
type: DSZ, layer: 3, pos: 698
type: DSZ, layer: 3, pos: 2340
type: DSZ, layer: 3, pos: 1733
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 1123
type: DSZ, layer: 3, pos: 1780
type: DSZ, layer: 3, pos: 598
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 647
type: DSZ, layer: 3, pos: 1679
type: DSZ, layer: 3, pos: 1457
type: DSZ, layer: 3, pos: 2469
type: DSZ, layer: 3, pos: 950

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 3, pos: 1977

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2533953, upper bound: 0.2429837
time: 3.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2526537, upper bound: 0.2449809
time: 3.44 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.2076020, -5.5223475, -6.2076020, -5.5223475, -0.3585200, 0.3623683
1: -6.1862726, -5.5502210, -6.1862726, -5.5502210, -0.3851202, 0.3870003
2: -9.9678802, -9.1544847, -9.9678802, -9.1544847, -0.3597100, 0.3578068
3: -8.3231497, -7.4329510, -8.3231497, -7.4329510, -0.4710572, 0.4610257
4: -7.5297031, -6.9073687, -7.5297031, -6.9073687, -0.3337759, 0.3356673
5: 3.6488411, 4.4215889, 3.6488411, 4.4215889, -0.3174634, 0.3173139
6: -4.4604273, -3.7868159, -4.4604273, -3.7868159, -0.5366099, 0.5417056
7: -9.7019253, -9.0353975, -9.7019253, -9.0353975, -0.3933887, 0.4025564
8: 0.4475598, 1.0476842, 0.4475598, 1.0476842, -0.4476194, 0.4493523
9: -4.8405838, -4.0761609, -4.8405838, -4.0761609, -0.4937906, 0.5029066

Time for backsubstitution: 21.82 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1977
type: DSZ, layer: 3, pos: 2144
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 312
type: DSZ, layer: 3, pos: 698
type: DSZ, layer: 3, pos: 2340
type: DSZ, layer: 3, pos: 1733
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 1123
type: DSZ, layer: 3, pos: 1780
type: DSZ, layer: 3, pos: 598
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 647
type: DSZ, layer: 3, pos: 1679
type: DSZ, layer: 3, pos: 1457
type: DSZ, layer: 3, pos: 2469
type: DSZ, layer: 3, pos: 950

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 3, pos: 1977

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2440946, upper bound: 0.2471836
time: 3.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2432427, upper bound: 0.2479197
time: 3.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.2076020, -5.5223475, -6.2076020, -5.5223475, -0.3599772, 0.3630275
1: -6.1862726, -5.5502210, -6.1862726, -5.5502210, -0.3843505, 0.3884180
2: -9.9678802, -9.1544847, -9.9678802, -9.1544847, -0.3589474, 0.3602484
3: -8.3231497, -7.4329510, -8.3231497, -7.4329510, -0.4611015, 0.4791670
4: -7.5297031, -6.9073687, -7.5297031, -6.9073687, -0.3329437, 0.3370503
5: 3.6488411, 4.4215889, 3.6488411, 4.4215889, -0.3247570, 0.3140006
6: -4.4604273, -3.7868159, -4.4604273, -3.7868159, -0.5416794, 0.5385671
7: -9.7019253, -9.0353975, -9.7019253, -9.0353975, -0.4020262, 0.4009638
8: 0.4475598, 1.0476842, 0.4475598, 1.0476842, -0.4488919, 0.4512379
9: -4.8405838, -4.0761609, -4.8405838, -4.0761609, -0.5021653, 0.5001633

Time for backsubstitution: 21.86 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1977
type: DSZ, layer: 3, pos: 2144
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 312
type: DSZ, layer: 3, pos: 698
type: DSZ, layer: 3, pos: 2340
type: DSZ, layer: 3, pos: 1733
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 1123
type: DSZ, layer: 3, pos: 1780
type: DSZ, layer: 3, pos: 598
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 647
type: DSZ, layer: 3, pos: 1679
type: DSZ, layer: 3, pos: 1457
type: DSZ, layer: 3, pos: 2469
type: DSZ, layer: 3, pos: 950

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 3, pos: 1977

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2491645, upper bound: 0.2420418
time: 3.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2484025, upper bound: 0.2429028
time: 3.39 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.2076020, -5.5223475, -6.2076020, -5.5223475, -0.3607757, 0.3601167
1: -6.1862726, -5.5502210, -6.1862726, -5.5502210, -0.3877337, 0.3844087
2: -9.9678802, -9.1544847, -9.9678802, -9.1544847, -0.3602427, 0.3572706
3: -8.3231497, -7.4329510, -8.3231497, -7.4329510, -0.4718468, 0.4602454
4: -7.5297031, -6.9073687, -7.5297031, -6.9073687, -0.3342506, 0.3352286
5: 3.6488411, 4.4215889, 3.6488411, 4.4215889, -0.3144243, 0.3203743
6: -4.4604273, -3.7868159, -4.4604273, -3.7868159, -0.5358694, 0.5424602
7: -9.7019253, -9.0353975, -9.7019253, -9.0353975, -0.3946314, 0.4013243
8: 0.4475598, 1.0476842, 0.4475598, 1.0476842, -0.4468539, 0.4501417
9: -4.8405838, -4.0761609, -4.8405838, -4.0761609, -0.4961042, 0.5006216

Time for backsubstitution: 21.90 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 56.53 + 554.74 = 611.26 seconds
