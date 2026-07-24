## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 0.81814518181
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (8.8957224, 11.2440281, 8.8957224, 11.2440281, -2.3483057, 2.3483057)
1: (-19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.8090019, 3.8090019)
2: (-3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.8001652, 2.8001652)
3: (-13.5667639, -10.0373669, -13.5667639, -10.0373669, -3.5293970, 3.5293970)
4: (-15.5752630, -11.9897022, -15.5752630, -11.9897022, -3.5616326, 3.5616322)
5: (-6.1198454, -3.7446783, -6.1198454, -3.7446783, -2.3524384, 2.3524384)
6: (-3.6156614, -1.3861105, -3.6156614, -1.3861105, -2.2295508, 2.2295508)
7: (-7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.8272722, 3.8272722)
8: (-2.8077574, -0.6163578, -2.8077574, -0.6163578, -2.1913996, 2.1913996)
9: (-9.4011059, -6.0042830, -9.4011059, -6.0042830, -3.2635217, 3.2635217)

## BASE Result
execution time: IAR + LP analysis = 14.95 + 32.59 = 47.54 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3552.46 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=2.031045436859131
rel_dist={0: [-1.052260884305138, 1.052260719977161]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.VERIFIED, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=1.8055601119995117
rel_dist={0: [-0.697324300994195, 0.6973245229667757]}

## Binary search (step 2) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=4, k_high=5, k_mid=4, eps_mid=0.0156250, abs_max=1.8807220458984375
rel_dist={0: [-0.8193859179682992, 0.8193858024795073]}

## Binary Search Result
Binary search time: 155.48 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.01171875


# Individual Split (IS_dual) starts
Time budget: 3396.98 seconds

## Binary search (step 0) starts
Candidate k: 8, corresponding eps: 0.0312500


## IAR start
Binary search (step 0): status=Status.ADV_EXAMPLE, k_low=4, k_high=12, k_mid=8, eps_mid=0.0312500, abs_max=None

## Binary search (step 1) starts
Candidate k: 5, corresponding eps: 0.0195312


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5734
type: B, layer: 1, pos: 5734
type: B, layer: 1, pos: 5758
type: A, layer: 1, pos: 5758
type: A, layer: 1, pos: 4568
type: B, layer: 1, pos: 4568
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 4631
type: B, layer: 1, pos: 4631
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5734

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9368811, upper bound: 0.9255753
time: 7.58 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9371562, upper bound: 0.9371556
time: 6.86 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 14.63 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 14.63
Output dim: 0, lower bound: -0.9368811, upper bound: 0.9255753
IS_A2, status: Status.UNKNOWN, split count: 1, time: 14.63
Output dim: 0, lower bound: -0.9371562, upper bound: 0.9371556

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 8.8992767, 11.2241116, 8.8968792, 11.2375221, -1.9451127, 1.9340291
1: -19.2491150, -15.4887924, -19.2528229, -15.4597158, -3.2554960, 3.2294002
2: -3.8380058, -1.0480461, -3.8398595, -1.0430375, -2.2690783, 2.2651603
3: -13.5404816, -10.0399275, -13.5581722, -10.0381985, -2.7663565, 2.7829022
4: -15.5738468, -12.0072260, -15.5748024, -11.9954376, -2.7447681, 2.7328663
5: -6.1104980, -3.7488604, -6.1167970, -3.7460337, -1.7490597, 1.7526865
6: -3.6106141, -1.4266734, -3.6140230, -1.3993807, -1.8958278, 1.8708725
7: -7.6639872, -3.9012482, -7.7056103, -3.8994467, -3.6229954, 3.6651621
8: -2.8004570, -0.6195345, -2.8053703, -0.6173859, -2.0319390, 2.0350895
9: -9.3933372, -6.0077019, -9.3986015, -6.0053906, -2.5166974, 2.5180402

Time for backsubstitution: 14.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5734
type: B, layer: 1, pos: 5758
type: A, layer: 1, pos: 5758
type: B, layer: 1, pos: 4568
type: A, layer: 1, pos: 4568
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 4631
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 4631
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 467

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5734

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9255730, upper bound: 0.9255753
time: 6.42 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9255730, upper bound: 0.9255733
time: 5.52 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 8.8560371, 11.2455463, 8.8957253, 11.2440147, -1.9940877, 1.9486775
1: -19.3369961, -15.4389858, -19.2545929, -15.4456406, -3.3572826, 3.2701659
2: -3.8489280, -1.0337003, -3.8407462, -1.0405868, -2.2832417, 2.2836885
3: -13.5858612, -9.9870853, -13.5667400, -10.0373697, -2.8097796, 2.8360057
4: -15.6134892, -11.9788074, -15.5752611, -11.9897175, -2.7957735, 2.7608109
5: -6.1280670, -3.7288983, -6.1198416, -3.7446809, -1.7710795, 1.7762592
6: -3.6841943, -1.3764181, -3.6156573, -1.3861318, -1.9605079, 1.9146419
7: -7.7513666, -3.7920871, -7.7258058, -3.8985827, -3.6987829, 3.7609487
8: -2.8119488, -0.5976191, -2.8077507, -0.6163588, -2.0435915, 2.0584197
9: -9.4152441, -5.9926920, -9.4010992, -6.0042877, -2.5392370, 2.5373368

Time for backsubstitution: 14.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5734
type: A, layer: 1, pos: 5758
type: B, layer: 1, pos: 5758
type: B, layer: 1, pos: 4568
type: A, layer: 1, pos: 4568
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 4631
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 4631
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 467

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5734

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9255730, upper bound: 0.9368811
time: 5.70 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9255730, upper bound: 0.9371558
time: 4.67 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 25.04 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 25.04
Output dim: 0, lower bound: -0.9255730, upper bound: 0.9255753
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 25.04
Output dim: 0, lower bound: -0.9255730, upper bound: 0.9255733
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 25.04
Output dim: 0, lower bound: -0.9255730, upper bound: 0.9368811
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 25.04
Output dim: 0, lower bound: -0.9255730, upper bound: 0.9371558

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: 8.8992767, 11.2241116, 8.8992767, 11.2241116, -1.9312983, 1.9312983
1: -19.2491150, -15.4887924, -19.2491150, -15.4887924, -3.2262344, 3.2262344
2: -3.8380058, -1.0480461, -3.8380058, -1.0480461, -2.2627678, 2.2627683
3: -13.5404816, -10.0399275, -13.5404816, -10.0399275, -2.7642727, 2.7642725
4: -15.5738468, -12.0072260, -15.5738468, -12.0072260, -2.7304854, 2.7304859
5: -6.1104980, -3.7488604, -6.1104980, -3.7488604, -1.7460227, 1.7460229
6: -3.6106141, -1.4266734, -3.6106141, -1.4266734, -1.8674126, 1.8674123
7: -7.6639872, -3.9012482, -7.6639872, -3.9012482, -3.6208267, 3.6208262
8: -2.8004570, -0.6195345, -2.8004570, -0.6195345, -2.0294657, 2.0294652
9: -9.3933372, -6.0077019, -9.3933372, -6.0077019, -2.5133400, 2.5133400

Time for backsubstitution: 14.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5758
type: B, layer: 1, pos: 5758
type: A, layer: 1, pos: 4568
type: B, layer: 1, pos: 4568
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 4631
type: B, layer: 1, pos: 4631
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5758

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9216506, upper bound: 0.9255716
time: 4.74 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9255725, upper bound: 0.9255707
time: 7.68 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 8.8992767, 11.2241116, 8.8561039, 11.2455378, -1.9530149, 1.9735117
1: -19.2491150, -15.4887924, -19.3364124, -15.4390049, -3.2744961, 3.3132710
2: -3.8380058, -1.0480461, -3.8488822, -1.0337423, -2.2778177, 2.2738740
3: -13.5404816, -10.0399275, -13.5852547, -9.9870911, -2.8082814, 2.8067062
4: -15.5738468, -12.0072260, -15.6131144, -11.9788332, -2.7572327, 2.7715960
5: -6.1104980, -3.7488604, -6.1280332, -3.7289164, -1.7663105, 1.7685339
6: -3.6106141, -1.4266734, -3.6841671, -1.3765457, -1.9180565, 1.9179314
7: -7.6639872, -3.9012482, -7.7512736, -3.7923779, -3.6945214, 3.7015629
8: -2.8004570, -0.6195345, -2.8119230, -0.5979195, -2.0498223, 2.0416269
9: -9.3933372, -6.0077019, -9.4149942, -5.9927363, -2.5282440, 2.5341012

Time for backsubstitution: 14.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5758
type: A, layer: 1, pos: 5758
type: A, layer: 1, pos: 4568
type: B, layer: 1, pos: 4568
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 4631
type: B, layer: 1, pos: 4631
type: B, layer: 1, pos: 920
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5758

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9255727, upper bound: 0.9216477
time: 5.24 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9255728, upper bound: 0.9255700
time: 4.42 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: 8.8561039, 11.2455378, 8.8992767, 11.2241116, -1.9735117, 1.9530149
1: -19.3364124, -15.4390049, -19.2491150, -15.4887924, -3.3132715, 3.2744966
2: -3.8488822, -1.0337423, -3.8380058, -1.0480461, -2.2738743, 2.2778182
3: -13.5852547, -9.9870911, -13.5404816, -10.0399275, -2.8067055, 2.8082814
4: -15.6131144, -11.9788332, -15.5738468, -12.0072260, -2.7715964, 2.7572327
5: -6.1280332, -3.7289164, -6.1104980, -3.7488604, -1.7685342, 1.7663105
6: -3.6841671, -1.3765457, -3.6106141, -1.4266734, -1.9179316, 1.9180565
7: -7.7512736, -3.7923779, -7.6639872, -3.9012482, -3.7015629, 3.6945207
8: -2.8119230, -0.5979195, -2.8004570, -0.6195345, -2.0416269, 2.0498223
9: -9.4149942, -5.9927363, -9.3933372, -6.0077019, -2.5341015, 2.5282440

Time for backsubstitution: 14.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5758
type: B, layer: 1, pos: 5758
type: B, layer: 1, pos: 4568
type: A, layer: 1, pos: 4568
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 4631
type: A, layer: 1, pos: 4631
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5758

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9216482, upper bound: 0.9368776
time: 5.21 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9255703, upper bound: 0.9368778
time: 6.44 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: 8.8560371, 11.2455463, 8.8560371, 11.2455463, -1.9550848, 1.9550848
1: -19.3369961, -15.4389858, -19.3369961, -15.4389858, -3.3173485, 3.3173485
2: -3.8489280, -1.0337003, -3.8489280, -1.0337003, -2.2857771, 2.2857778
3: -13.5858612, -9.9870853, -13.5858612, -9.9870853, -2.8506589, 2.8506587
4: -15.6134892, -11.9788074, -15.6134892, -11.9788074, -2.8024750, 2.8024755
5: -6.1280670, -3.7288983, -6.1280670, -3.7288983, -1.7787466, 1.7787464
6: -3.6841943, -1.3764181, -3.6841943, -1.3764181, -1.9475718, 1.9475718
7: -7.7513666, -3.7920871, -7.7513666, -3.7920871, -3.7690306, 3.7690306
8: -2.8119488, -0.5976191, -2.8119488, -0.5976191, -2.0537682, 2.0537682
9: -9.4152441, -5.9926920, -9.4152441, -5.9926920, -2.5497146, 2.5497146

Time for backsubstitution: 14.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5758
type: A, layer: 1, pos: 5758
type: A, layer: 1, pos: 4568
type: B, layer: 1, pos: 4568
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 4631
type: B, layer: 1, pos: 4631
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5758

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9255705, upper bound: 0.9332338
time: 5.49 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9255704, upper bound: 0.9371543
time: 5.41 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 25.60 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 25.60
Output dim: 0, lower bound: -0.9216506, upper bound: 0.9255716
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 25.60
Output dim: 0, lower bound: -0.9255725, upper bound: 0.9255707
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 25.60
Output dim: 0, lower bound: -0.9255727, upper bound: 0.9216477
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 25.60
Output dim: 0, lower bound: -0.9255728, upper bound: 0.9255700
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 25.60
Output dim: 0, lower bound: -0.9216482, upper bound: 0.9368776
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 25.60
Output dim: 0, lower bound: -0.9255703, upper bound: 0.9368778
IS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 25.60
Output dim: 0, lower bound: -0.9255705, upper bound: 0.9332338
IS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 25.60
Output dim: 0, lower bound: -0.9255704, upper bound: 0.9371543

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: 8.9226303, 11.2084036, 8.9076347, 11.2235813, -1.9056969, 1.9059329
1: -19.2102623, -15.5165377, -19.2455730, -15.5001345, -3.1743469, 3.1715841
2: -3.8174486, -1.0676342, -3.8325396, -1.0502604, -2.2344575, 2.2452157
3: -13.5026731, -10.0723238, -13.5301113, -10.0433826, -2.7173629, 2.6696553
4: -15.5580549, -12.0238962, -15.5727167, -12.0128298, -2.7122183, 2.7094779
5: -6.1050787, -3.7678983, -6.1097937, -3.7518001, -1.7380257, 1.7228470
6: -3.6007564, -1.4359775, -3.6075468, -1.4283054, -1.8649893, 1.8515425
7: -7.6213474, -3.9494665, -7.6600361, -3.9180751, -3.5455689, 3.5709977
8: -2.7791033, -0.6302538, -2.7951488, -0.6202784, -2.0073509, 2.0137377
9: -9.3696842, -6.0306206, -9.3854742, -6.0096259, -2.4863997, 2.4914360

Time for backsubstitution: 14.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5758
type: B, layer: 1, pos: 4568
type: A, layer: 1, pos: 4568
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 4631
type: A, layer: 1, pos: 4631
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5758

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9216505, upper bound: 0.9216528
time: 5.72 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9216505, upper bound: 0.9255728
time: 5.04 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: 8.8992863, 11.2241116, 8.8992815, 11.2241116, -1.9210067, 1.9312932
1: -19.2491074, -15.4888048, -19.2491131, -15.4887972, -3.2262230, 3.2074275
2: -3.8379967, -1.0480481, -3.8380029, -1.0480469, -2.2572317, 2.2627614
3: -13.5404644, -10.0399332, -13.5404730, -10.0399323, -2.7489781, 2.7642627
4: -15.5738430, -12.0072346, -15.5738468, -12.0072279, -2.7285814, 2.7226095
5: -6.1104951, -3.7488642, -6.1104960, -3.7488623, -1.7454739, 1.7452672
6: -3.6106095, -1.4266739, -3.6106119, -1.4266739, -1.8641143, 1.8708379
7: -7.6639819, -3.9012637, -7.6639843, -3.9012532, -3.6208134, 3.6050081
8: -2.8004494, -0.6195374, -2.8004546, -0.6195374, -2.0231581, 2.0294604
9: -9.3933258, -6.0077038, -9.3933334, -6.0077019, -2.4951072, 2.5133328

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4568
type: B, layer: 1, pos: 5758
type: B, layer: 1, pos: 4568
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 4631
type: B, layer: 1, pos: 920
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 4631
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 467

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4568

## Relational analysis of IS_A1_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9244122, upper bound: 0.9255712
time: 4.98 seconds

## Relational analysis of IS_A1_B1_A2_A2

### Relational analysis result of IS_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9255687, upper bound: 0.9255688
time: 8.11 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: 8.9076347, 11.2235813, 8.8823147, 11.2298260, -1.9276481, 1.9451432
1: -19.2455730, -15.5001345, -19.2950668, -15.4667673, -3.2198334, 3.2541461
2: -3.8325396, -1.0502604, -3.8283353, -1.0545851, -2.2585359, 2.2455611
3: -13.5301113, -10.0433826, -13.5473003, -10.0233955, -2.7154498, 2.7596707
4: -15.5727167, -12.0128298, -15.5946226, -11.9955750, -2.7359314, 2.7500353
5: -6.1097937, -3.7518001, -6.1168470, -3.7478752, -1.7432153, 1.7519627
6: -3.6075468, -1.4283054, -3.6684361, -1.3857559, -1.9022593, 1.9086778
7: -7.6600361, -3.9180751, -7.7089362, -3.8411403, -3.6441293, 3.6264396
8: -2.7951488, -0.6202784, -2.7906046, -0.6106958, -2.0320129, 2.0195498
9: -9.3854742, -6.0096259, -9.3911753, -6.0158834, -2.5060248, 2.5069771

Time for backsubstitution: 14.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5758
type: A, layer: 1, pos: 4568
type: B, layer: 1, pos: 4568
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 920
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 4631
type: B, layer: 1, pos: 4631
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5758

## Relational analysis of IS_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9329543, upper bound: 0.9216485
time: 5.26 seconds

## Relational analysis of IS_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9329543, upper bound: 0.9216482
time: 7.38 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: 8.8992815, 11.2241116, 8.8561115, 11.2455378, -1.9530101, 1.9616427
1: -19.2491131, -15.4887972, -19.3364067, -15.4390163, -3.2556887, 3.2948916
2: -3.8380029, -1.0480469, -3.8488741, -1.0337453, -2.2778091, 2.2683599
3: -13.5404730, -10.0399323, -13.5852375, -9.9870930, -2.7874570, 2.7914124
4: -15.5738468, -12.0072279, -15.6131134, -11.9788380, -2.7491922, 2.7696414
5: -6.1104960, -3.7488623, -6.1280298, -3.7289200, -1.7655549, 1.7679856
6: -3.6106119, -1.4266739, -3.6841607, -1.3765495, -1.9214816, 1.9129515
7: -7.6639843, -3.9012532, -7.7512708, -3.7923930, -3.6687293, 3.7015491
8: -2.8004546, -0.6195374, -2.8119140, -0.5979204, -2.0498161, 2.0353198
9: -9.3933334, -6.0077019, -9.4149847, -5.9927363, -2.5282369, 2.5156217

Time for backsubstitution: 14.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4568
type: A, layer: 1, pos: 5758
type: A, layer: 1, pos: 4568
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 4631
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 920
type: A, layer: 1, pos: 4631
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 467

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4568

## Relational analysis of IS_A1_B2_B2_B1

### Relational analysis result of IS_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9368737, upper bound: 0.9244104
time: 5.10 seconds

## Relational analysis of IS_A1_B2_B2_B2

### Relational analysis result of IS_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9368737, upper bound: 0.9255688
time: 6.09 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: 8.8823147, 11.2298260, 8.9076347, 11.2235813, -1.9451432, 1.9276481
1: -19.2950668, -15.4667673, -19.2455730, -15.5001345, -3.2541466, 3.2198334
2: -3.8283353, -1.0545851, -3.8325396, -1.0502604, -2.2455611, 2.2585356
3: -13.5473003, -10.0233955, -13.5301113, -10.0433826, -2.7596707, 2.7154498
4: -15.5946226, -11.9955750, -15.5727167, -12.0128298, -2.7500353, 2.7359314
5: -6.1168470, -3.7478752, -6.1097937, -3.7518001, -1.7519627, 1.7432151
6: -3.6684361, -1.3857559, -3.6075468, -1.4283054, -1.9086781, 1.9022596
7: -7.7089362, -3.8411403, -7.6600361, -3.9180751, -3.6264405, 3.6441293
8: -2.7906046, -0.6106958, -2.7951488, -0.6202784, -2.0195494, 2.0320129
9: -9.3911753, -6.0158834, -9.3854742, -6.0096259, -2.5069776, 2.5060248

Time for backsubstitution: 14.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5758
type: B, layer: 1, pos: 4568
type: A, layer: 1, pos: 4568
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 4631
type: A, layer: 1, pos: 4631
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5758

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9216482, upper bound: 0.9329565
time: 7.33 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9216482, upper bound: 0.9368798
time: 4.91 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: 8.8561115, 11.2455378, 8.8992815, 11.2241116, -1.9616427, 1.9530101
1: -19.3364067, -15.4390163, -19.2491131, -15.4887972, -3.2948909, 3.2556882
2: -3.8488741, -1.0337453, -3.8380029, -1.0480469, -2.2683592, 2.2778099
3: -13.5852375, -9.9870930, -13.5404730, -10.0399323, -2.7914128, 2.7874568
4: -15.6131134, -11.9788380, -15.5738468, -12.0072279, -2.7696409, 2.7491927
5: -6.1280298, -3.7289200, -6.1104960, -3.7488623, -1.7679853, 1.7655551
6: -3.6841607, -1.3765495, -3.6106119, -1.4266739, -1.9129515, 1.9214818
7: -7.7512708, -3.7923930, -7.6639843, -3.9012532, -3.7015495, 3.6687295
8: -2.8119140, -0.5979204, -2.8004546, -0.6195374, -2.0353193, 2.0498166
9: -9.4149847, -5.9927363, -9.3933334, -6.0077019, -2.5156217, 2.5282369

Time for backsubstitution: 14.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4568
type: B, layer: 1, pos: 5758
type: B, layer: 1, pos: 4568
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 4631
type: B, layer: 1, pos: 920
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 4631
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 467

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4568

## Relational analysis of IS_A2_B1_A2_A1

### Relational analysis result of IS_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9244100, upper bound: 0.9368740
time: 4.94 seconds

## Relational analysis of IS_A2_B1_A2_A2

### Relational analysis result of IS_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9255664, upper bound: 0.9368736
time: 7.14 seconds

## BFS IS instance: IS_A2_B2_B1

### Backsubstitution after applying IS history:
0: 8.8650475, 11.2450094, 8.8822489, 11.2298317, -1.9291039, 1.9268403
1: -19.3328991, -15.4503288, -19.2956505, -15.4667501, -3.2619705, 3.2624116
2: -3.8434677, -1.0362083, -3.8283813, -1.0545433, -2.2663994, 2.2570915
3: -13.5754700, -9.9914103, -13.5479069, -10.0233917, -2.7561150, 2.8025403
4: -15.6117458, -11.9844160, -15.5949945, -11.9955511, -2.7804184, 2.7806025
5: -6.1260662, -3.7318182, -6.1168823, -3.7478588, -1.7536731, 1.7621853
6: -3.6798124, -1.3780277, -3.6684625, -1.3856263, -1.9305329, 1.9396629
7: -7.7474837, -3.8090475, -7.7090282, -3.8408487, -3.7186508, 3.6846962
8: -2.8066440, -0.5988188, -2.7906318, -0.6103935, -2.0359578, 2.0312238
9: -9.4073277, -5.9946589, -9.3914280, -6.0158410, -2.5273132, 2.5225389

Time for backsubstitution: 14.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5758
type: A, layer: 1, pos: 4568
type: B, layer: 1, pos: 4568
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 4631
type: B, layer: 1, pos: 920
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 4631
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5758

## Relational analysis of IS_A2_B2_B1_A1

### Relational analysis result of IS_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9218558, upper bound: 0.9332314
time: 5.06 seconds

## Relational analysis of IS_A2_B2_B1_A2

### Relational analysis result of IS_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9218558, upper bound: 0.9332311
time: 7.07 seconds

## BFS IS instance: IS_A2_B2_B2

### Backsubstitution after applying IS history:
0: 8.8560410, 11.2455463, 8.8560467, 11.2455463, -1.9550805, 1.9447932
1: -19.3369942, -15.4389935, -19.3369923, -15.4390011, -3.2985411, 3.3173380
2: -3.8489246, -1.0337021, -3.8489184, -1.0337025, -2.2857718, 2.2802641
3: -13.5858517, -9.9870892, -13.5858421, -9.9870930, -2.8298340, 2.8289299
4: -15.6134882, -11.9788074, -15.6134863, -11.9788151, -2.7945023, 2.8005195
5: -6.1280661, -3.7288990, -6.1280642, -3.7289014, -1.7779899, 1.7781975
6: -3.6841915, -1.3764193, -3.6841879, -1.3764200, -1.9512401, 1.9445167
7: -7.7513647, -3.7920938, -7.7513618, -3.7921023, -3.7433434, 3.7411885
8: -2.8119469, -0.5976200, -2.8119426, -0.5976200, -2.0537639, 2.0474591
9: -9.4152403, -5.9926925, -9.4152336, -5.9926934, -2.5497065, 2.5312343

Time for backsubstitution: 14.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4568
type: A, layer: 1, pos: 5758
type: A, layer: 1, pos: 4568
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 4631
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 920
type: A, layer: 1, pos: 4631
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 467

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 4568

## Relational analysis of IS_A2_B2_B2_B1

### Relational analysis result of IS_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9257742, upper bound: 0.9359930
time: 4.61 seconds

## Relational analysis of IS_A2_B2_B2_B2

### Relational analysis result of IS_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9257742, upper bound: 0.9371485
time: 4.43 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 23.58 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 23.58
Output dim: 0, lower bound: -0.9216505, upper bound: 0.9216528
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.58
Output dim: 0, lower bound: -0.9216505, upper bound: 0.9255728
IS_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 23.58
Output dim: 0, lower bound: -0.9244122, upper bound: 0.9255712
IS_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 23.58
Output dim: 0, lower bound: -0.9255687, upper bound: 0.9255688
IS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 23.58
Output dim: 0, lower bound: -0.9329543, upper bound: 0.9216485
IS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 23.58
Output dim: 0, lower bound: -0.9329543, upper bound: 0.9216482
IS_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 23.58
Output dim: 0, lower bound: -0.9368737, upper bound: 0.9244104
IS_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 23.58
Output dim: 0, lower bound: -0.9368737, upper bound: 0.9255688
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 23.58
Output dim: 0, lower bound: -0.9216482, upper bound: 0.9329565
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.58
Output dim: 0, lower bound: -0.9216482, upper bound: 0.9368798
IS_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 23.58
Output dim: 0, lower bound: -0.9244100, upper bound: 0.9368740
IS_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 23.58
Output dim: 0, lower bound: -0.9255664, upper bound: 0.9368736
IS_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 23.58
Output dim: 0, lower bound: -0.9218558, upper bound: 0.9332314
IS_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 23.58
Output dim: 0, lower bound: -0.9218558, upper bound: 0.9332311
IS_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 23.58
Output dim: 0, lower bound: -0.9257742, upper bound: 0.9359930
IS_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 23.58
Output dim: 0, lower bound: -0.9257742, upper bound: 0.9371485

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 8.9226303, 11.2084036, 8.9226303, 11.2084036, -1.8891759, 1.8891757
1: -19.2102623, -15.5165377, -19.2102623, -15.5165377, -3.1369638, 3.1369638
2: -3.8174486, -1.0676342, -3.8174486, -1.0676342, -2.2206316, 2.2206314
3: -13.5026731, -10.0723238, -13.5026731, -10.0723238, -2.6382785, 2.6382785
4: -15.5580549, -12.0238962, -15.5580549, -12.0238962, -2.6972389, 2.6972384
5: -6.1050787, -3.7678983, -6.1050787, -3.7678983, -1.7181149, 1.7181151
6: -3.6007564, -1.4359775, -3.6007564, -1.4359775, -1.8532066, 1.8532064
7: -7.6213474, -3.9494665, -7.6213474, -3.9494665, -3.5167370, 3.5167365
8: -2.7791033, -0.6302538, -2.7791033, -0.6302538, -1.9976597, 1.9976602
9: -9.3696842, -6.0306206, -9.3696842, -6.0306206, -2.4776673, 2.4776673

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4568
type: B, layer: 1, pos: 4568
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 920
type: A, layer: 1, pos: 4631
type: B, layer: 1, pos: 4631
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4568

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9204908, upper bound: 0.9218099
time: 5.97 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9216469, upper bound: 0.9218097
time: 7.07 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 8.9226303, 11.2084036, 8.8993397, 11.2241116, -1.9063096, 1.9141052
1: -19.2102623, -15.5165377, -19.2490654, -15.4888048, -3.1855011, 3.1753469
2: -3.8174486, -1.0676342, -3.8379967, -1.0480732, -2.2372437, 2.2456698
3: -13.5026731, -10.0723238, -13.5404644, -10.0400038, -2.7199035, 2.6826887
4: -15.5580549, -12.0238962, -15.5737972, -12.0072346, -2.7161355, 2.7093649
5: -6.1050787, -3.7678983, -6.1103830, -3.7488642, -1.7390842, 1.7234454
6: -3.6007564, -1.4359775, -3.6104999, -1.4266739, -1.8622746, 1.8527012
7: -7.6213474, -3.9494665, -7.6639819, -3.9012723, -3.5634117, 3.5740423
8: -2.7791033, -0.6302538, -2.8004494, -0.6195745, -2.0081902, 2.0190234
9: -9.3696842, -6.0306206, -9.3933258, -6.0077071, -2.4886537, 2.5018206

Time for backsubstitution: 14.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4568
type: B, layer: 1, pos: 4568
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 920
type: A, layer: 1, pos: 4631
type: B, layer: 1, pos: 4631
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4568

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9204909, upper bound: 0.9255691
time: 4.85 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9216470, upper bound: 0.9255686
time: 5.00 seconds

## BFS IS instance: IS_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: 8.9150639, 11.2163744, 8.9031773, 11.2239304, -1.9044333, 1.9168584
1: -19.2305946, -15.5108891, -19.2444916, -15.4897375, -3.2056580, 3.1804719
2: -3.8190370, -1.0579590, -3.8352203, -1.0495210, -2.2319927, 2.2503178
3: -13.5160532, -10.0603151, -13.5352039, -10.0406742, -2.7155337, 2.7407808
4: -15.5616665, -12.0215187, -15.5731831, -12.0119343, -2.7112489, 2.7038846
5: -6.1050496, -3.7642772, -6.1088099, -3.7499180, -1.7391665, 1.7267101
6: -3.5876651, -1.4449077, -3.6034403, -1.4280982, -1.8398590, 1.8443167
7: -7.6383028, -3.9263783, -7.6616168, -3.9101861, -3.5859032, 3.5780840
8: -2.7905188, -0.6271248, -2.7984962, -0.6200733, -2.0124645, 2.0185027
9: -9.3845539, -6.0175924, -9.3915291, -6.0086722, -2.4830880, 2.4993336

Time for backsubstitution: 14.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5758
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 4568
type: B, layer: 1, pos: 920
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 4631
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 4631
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5758

## Relational analysis of IS_A1_B1_A2_A1_B1

### Relational analysis result of IS_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9244122, upper bound: 0.9216493
time: 4.92 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2

### Relational analysis result of IS_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9244125, upper bound: 0.9216493
time: 4.81 seconds

## BFS IS instance: IS_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: 8.8992996, 11.2241106, 8.8992882, 11.2241106, -1.9176445, 1.9312866
1: -19.2490959, -15.4888077, -19.2491093, -15.4887981, -3.2217493, 3.2074184
2: -3.8379884, -1.0480522, -3.8379998, -1.0480486, -2.2572575, 2.2626991
3: -13.5404482, -10.0399361, -13.5404692, -10.0399342, -2.7467775, 2.7626114
4: -15.5738449, -12.0072498, -15.5738459, -12.0072336, -2.7278237, 2.7197771
5: -6.1104894, -3.7488661, -6.1104937, -3.7488627, -1.7425709, 1.7452626
6: -3.6105895, -1.4266779, -3.6106052, -1.4266760, -1.8483891, 1.8708272
7: -7.6639757, -3.9012990, -7.6639814, -3.9012659, -3.6207981, 3.5919914
8: -2.8004432, -0.6195374, -2.8004522, -0.6195374, -2.0225153, 2.0294576
9: -9.3933201, -6.0077062, -9.3933296, -6.0077047, -2.4948640, 2.5131490

Time for backsubstitution: 14.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 5758
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 4631
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 920
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 4631
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 4568
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 467

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 820

## Relational analysis of IS_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 820

## Relational analysis of IS_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6137

## Relational analysis of IS_A1_B1_A2_A2_B1

### Relational analysis result of IS_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9246097, upper bound: 0.9248244
time: 4.97 seconds

## Relational analysis of IS_A1_B1_A2_A2_B2

### Relational analysis result of IS_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9255686, upper bound: 0.9255711
time: 4.75 seconds

## BFS IS instance: IS_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: 8.9226303, 11.2084036, 8.8823147, 11.2298260, -1.9108906, 1.9286222
1: -19.2102623, -15.5165377, -19.2950668, -15.4667673, -3.1852131, 3.2188449
2: -3.8174486, -1.0676342, -3.8283353, -1.0545851, -2.2339511, 2.2317631
3: -13.5026731, -10.0723238, -13.5473003, -10.0233955, -2.6776114, 2.6790771
4: -15.5580549, -12.0238962, -15.5946226, -11.9955750, -2.7235565, 2.7350564
5: -6.1050787, -3.7678983, -6.1168470, -3.7478752, -1.7384832, 1.7320304
6: -3.6007564, -1.4359775, -3.6684361, -1.3857559, -1.9039893, 1.8956738
7: -7.6213474, -3.9494665, -7.7089362, -3.8411403, -3.5874968, 3.5976076
8: -2.7791033, -0.6302538, -2.7906046, -0.6106958, -2.0159359, 2.0098600
9: -9.3696842, -6.0306206, -9.3911753, -6.0158834, -2.4922562, 2.4983349

Time for backsubstitution: 14.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4568
type: B, layer: 1, pos: 4568
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 920
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 4631
type: B, layer: 1, pos: 4631
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4568

## Relational analysis of IS_A1_B2_B1_A1_A1

### Relational analysis result of IS_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9319574, upper bound: 0.9216448
time: 7.11 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2

### Relational analysis result of IS_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9331133, upper bound: 0.9216448
time: 15.52 seconds

## BFS IS instance: IS_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: 8.8993397, 11.2241116, 8.8823147, 11.2298260, -1.9358201, 1.9346869
1: -19.2490654, -15.4888048, -19.2950668, -15.4667673, -3.2235966, 3.2581022
2: -3.8379967, -1.0480732, -3.8283353, -1.0545851, -2.2589898, 2.2483478
3: -13.5404644, -10.0400038, -13.5473003, -10.0233955, -2.7143402, 2.7622106
4: -15.5737972, -12.0072346, -15.5946226, -11.9955750, -2.7358189, 2.7539530
5: -6.1103830, -3.7488642, -6.1168470, -3.7478752, -1.7438133, 1.7530210
6: -3.6104999, -1.4266739, -3.6684361, -1.3857559, -1.9034181, 1.9052432
7: -7.6639819, -3.9012723, -7.7089362, -3.8411403, -3.6189656, 3.6442823
8: -2.8004494, -0.6195745, -2.7906046, -0.6106958, -2.0372992, 2.0203896
9: -9.3933258, -6.0077071, -9.3911753, -6.0158834, -2.5164094, 2.5092316

Time for backsubstitution: 14.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4568
type: A, layer: 1, pos: 4568
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 4631
type: B, layer: 1, pos: 920
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 4631
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4568

## Relational analysis of IS_A1_B2_B1_A2_B1

### Relational analysis result of IS_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9331134, upper bound: 0.9204884
time: 4.68 seconds

## Relational analysis of IS_A1_B2_B1_A2_B2

### Relational analysis result of IS_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9331143, upper bound: 0.9216470
time: 5.06 seconds

## BFS IS instance: IS_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: 8.9031773, 11.2239304, 8.8733425, 11.2377968, -1.9385738, 1.9435911
1: -19.2444916, -15.4897375, -19.3164940, -15.4611816, -3.2289696, 3.2730238
2: -3.8352203, -1.0495210, -3.8299320, -1.0443202, -2.2645555, 2.2431266
3: -13.5352039, -10.0406742, -13.5608139, -10.0094624, -2.7616458, 2.7580013
4: -15.5731831, -12.0119343, -15.5995779, -11.9931097, -2.7304187, 2.7505703
5: -6.1088099, -3.7499180, -6.1195378, -3.7443066, -1.7470422, 1.7572227
6: -3.6034403, -1.4280982, -3.6581645, -1.3947475, -1.8949900, 1.8858271
7: -7.6616168, -3.9101861, -7.7256541, -3.8177722, -3.6413708, 3.6669879
8: -2.7984962, -0.6200733, -2.8020349, -0.6065578, -2.0377717, 2.0246820
9: -9.3915291, -6.0086722, -9.4062357, -6.0027637, -2.5140510, 2.5036030

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5758
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 4568
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 4631
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 4631
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5758

## Relational analysis of IS_A1_B2_B2_B1_A1

### Relational analysis result of IS_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9329507, upper bound: 0.9244121
time: 5.56 seconds

## Relational analysis of IS_A1_B2_B2_B1_A2

### Relational analysis result of IS_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9329507, upper bound: 0.9204887
time: 4.74 seconds

## BFS IS instance: IS_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: 8.8992882, 11.2241106, 8.8561239, 11.2455378, -1.9530039, 1.9557359
1: -19.2491093, -15.4887981, -19.3363972, -15.4390182, -3.2556801, 3.2853711
2: -3.8379998, -1.0480486, -3.8488648, -1.0337480, -2.2777443, 2.2683854
3: -13.5404692, -10.0399342, -13.5852222, -9.9870977, -2.7767811, 2.7892118
4: -15.5738459, -12.0072336, -15.6131115, -11.9788513, -2.7463589, 2.7688828
5: -6.1104937, -3.7488627, -6.1280246, -3.7289233, -1.7655501, 1.7650821
6: -3.6106052, -1.4266760, -3.6841428, -1.3765540, -1.9214711, 1.8956721
7: -7.6639814, -3.9012659, -7.7512631, -3.7924292, -3.6526880, 3.7015338
8: -2.8004522, -0.6195374, -2.8119082, -0.5979223, -2.0498142, 2.0346766
9: -9.3933296, -6.0077047, -9.4149799, -5.9927387, -2.5280542, 2.5153787

Time for backsubstitution: 14.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 5758
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 4631
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 920
type: A, layer: 1, pos: 4631
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 4568
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 467

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 820

## Relational analysis of IS_A1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 820

## Relational analysis of IS_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5758

## Relational analysis of IS_A1_B2_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9329507, upper bound: 0.9255661
time: 11.77 seconds

## Relational analysis of IS_A1_B2_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9329517, upper bound: 0.9216470
time: 5.86 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 8.8823147, 11.2298260, 8.9226303, 11.2084036, -1.9286222, 1.9108906
1: -19.2950668, -15.4667673, -19.2102623, -15.5165377, -3.2188454, 3.1852131
2: -3.8283353, -1.0545851, -3.8174486, -1.0676342, -2.2317638, 2.2339509
3: -13.5473003, -10.0233955, -13.5026731, -10.0723238, -2.6790771, 2.6776111
4: -15.5946226, -11.9955750, -15.5580549, -12.0238962, -2.7350559, 2.7235565
5: -6.1168470, -3.7478752, -6.1050787, -3.7678983, -1.7320304, 1.7384830
6: -3.6684361, -1.3857559, -3.6007564, -1.4359775, -1.8956740, 1.9039896
7: -7.7089362, -3.8411403, -7.6213474, -3.9494665, -3.5976067, 3.5874963
8: -2.7906046, -0.6106958, -2.7791033, -0.6302538, -2.0098610, 2.0159359
9: -9.3911753, -6.0158834, -9.3696842, -6.0306206, -2.4983349, 2.4922562

Time for backsubstitution: 14.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4568
type: A, layer: 1, pos: 4568
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 4631
type: A, layer: 1, pos: 4631
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4568

## Relational analysis of IS_A2_B1_A1_B1_B1

### Relational analysis result of IS_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9216448, upper bound: 0.9319576
time: 6.53 seconds

## Relational analysis of IS_A2_B1_A1_B1_B2

### Relational analysis result of IS_A2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9216447, upper bound: 0.9331156
time: 4.67 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 8.8823147, 11.2298260, 8.8993397, 11.2241116, -1.9346874, 1.9358201
1: -19.2950668, -15.4667673, -19.2490654, -15.4888048, -3.2581024, 3.2235961
2: -3.8283353, -1.0545851, -3.8379967, -1.0480732, -2.2483473, 2.2589893
3: -13.5473003, -10.0233955, -13.5404644, -10.0400038, -2.7622113, 2.7143397
4: -15.5946226, -11.9955750, -15.5737972, -12.0072346, -2.7539525, 2.7358189
5: -6.1168470, -3.7478752, -6.1103830, -3.7488642, -1.7530212, 1.7438138
6: -3.6684361, -1.3857559, -3.6104999, -1.4266739, -1.9052429, 1.9034183
7: -7.7089362, -3.8411403, -7.6639819, -3.9012723, -3.6442814, 3.6189656
8: -2.7906046, -0.6106958, -2.8004494, -0.6195745, -2.0203896, 2.0372992
9: -9.3911753, -6.0158834, -9.3933258, -6.0077071, -2.5092316, 2.5164094

Time for backsubstitution: 14.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4568
type: B, layer: 1, pos: 4568
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 4631
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 4631
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4568

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9204885, upper bound: 0.9368760
time: 4.49 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9216448, upper bound: 0.9368738
time: 4.71 seconds

## BFS IS instance: IS_A2_B1_A2_A1

### Backsubstitution after applying IS history:
0: 8.8733425, 11.2377968, 8.9031773, 11.2239304, -1.9435911, 1.9385738
1: -19.3164940, -15.4611816, -19.2444916, -15.4897375, -3.2730241, 3.2289696
2: -3.8299320, -1.0443202, -3.8352203, -1.0495210, -2.2431269, 2.2645559
3: -13.5608139, -10.0094624, -13.5352039, -10.0406742, -2.7580009, 2.7616458
4: -15.5995779, -11.9931097, -15.5731831, -12.0119343, -2.7505708, 2.7304182
5: -6.1195378, -3.7443066, -6.1088099, -3.7499180, -1.7572224, 1.7470422
6: -3.6581645, -1.3947475, -3.6034403, -1.4280982, -1.8858271, 1.8949897
7: -7.7256541, -3.8177722, -7.6616168, -3.9101861, -3.6669884, 3.6413705
8: -2.8020349, -0.6065578, -2.7984962, -0.6200733, -2.0246811, 2.0377722
9: -9.4062357, -6.0027637, -9.3915291, -6.0086722, -2.5036035, 2.5140510

Time for backsubstitution: 14.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5758
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 4568
type: B, layer: 1, pos: 920
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 4631
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 4631
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5758

## Relational analysis of IS_A2_B1_A2_A1_B1

### Relational analysis result of IS_A2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9244098, upper bound: 0.9329510
time: 4.73 seconds

## Relational analysis of IS_A2_B1_A2_A1_B2

### Relational analysis result of IS_A2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9244101, upper bound: 0.9329510
time: 4.51 seconds

## BFS IS instance: IS_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: 8.8561239, 11.2455378, 8.8992882, 11.2241106, -1.9557357, 1.9530044
1: -19.3363972, -15.4390182, -19.2491093, -15.4887981, -3.2853708, 3.2556806
2: -3.8488648, -1.0337480, -3.8379998, -1.0480486, -2.2683849, 2.2777448
3: -13.5852222, -9.9870977, -13.5404692, -10.0399342, -2.7892122, 2.7767808
4: -15.6131115, -11.9788513, -15.5738459, -12.0072336, -2.7688832, 2.7463589
5: -6.1280246, -3.7289233, -6.1104937, -3.7488627, -1.7650819, 1.7655499
6: -3.6841428, -1.3765540, -3.6106052, -1.4266760, -1.8956718, 1.9214711
7: -7.7512631, -3.7924292, -7.6639814, -3.9012659, -3.7015343, 3.6526883
8: -2.8119082, -0.5979223, -2.8004522, -0.6195374, -2.0346766, 2.0498142
9: -9.4149799, -5.9927387, -9.3933296, -6.0077047, -2.5153790, 2.5280542

Time for backsubstitution: 14.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 5758
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 4631
type: B, layer: 1, pos: 920
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 4631
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 4568
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 467

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 820

## Relational analysis of IS_A2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 820

## Relational analysis of IS_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5758

## Relational analysis of IS_A2_B1_A2_A2_B1

### Relational analysis result of IS_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9255662, upper bound: 0.9329505
time: 7.02 seconds

## Relational analysis of IS_A2_B1_A2_A2_B2

### Relational analysis result of IS_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9255666, upper bound: 0.9329503
time: 8.13 seconds

## BFS IS instance: IS_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: 8.8822489, 11.2298317, 8.8822489, 11.2298317, -1.9103179, 1.9103181
1: -19.2956505, -15.4667501, -19.2956505, -15.4667501, -3.2250080, 3.2250080
2: -3.8283813, -1.0545433, -3.8283813, -1.0545433, -2.2417946, 2.2417939
3: -13.5479069, -10.0233917, -13.5479069, -10.0233917, -2.7183452, 2.7183454
4: -15.5949945, -11.9955511, -15.5949945, -11.9955511, -2.7655058, 2.7655067
5: -6.1168823, -3.7478588, -6.1168823, -3.7478588, -1.7423239, 1.7423236
6: -3.6684625, -1.3856263, -3.6684625, -1.3856263, -1.9279461, 1.9279459
7: -7.7090282, -3.8408487, -7.7090282, -3.8408487, -3.6620541, 3.6620545
8: -2.7906318, -0.6103935, -2.7906318, -0.6103935, -2.0199199, 2.0199199
9: -9.3914280, -6.0158410, -9.3914280, -6.0158410, -2.5136313, 2.5136311

Time for backsubstitution: 14.45 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=4, k_high=7, k_mid=5, eps_mid=0.0195312, abs_max=1.955883502960205
rel_dist={0: [-0.9371681487262702, 0.9371704841879342]}

## Binary search (step 2) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5734
type: B, layer: 1, pos: 5734
type: B, layer: 1, pos: 5758
type: A, layer: 1, pos: 5758
type: B, layer: 1, pos: 4568
type: A, layer: 1, pos: 4568
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 4631
type: A, layer: 1, pos: 4631
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5734

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8191166, upper bound: 0.8101018
time: 4.88 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8193768, upper bound: 0.8193757
time: 4.75 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 9.81 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 9.81
Output dim: 0, lower bound: -0.8191166, upper bound: 0.8101018
IS_A2, status: Status.UNKNOWN, split count: 1, time: 9.81
Output dim: 0, lower bound: -0.8193768, upper bound: 0.8193757

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 8.8992767, 11.2241116, 8.8972569, 11.2353973, -1.8677664, 1.8584299
1: -19.2491150, -15.4887924, -19.2522411, -15.4643192, -3.1295767, 3.1076169
2: -3.8380058, -1.0480461, -3.8395674, -1.0438364, -2.1883712, 2.1850648
3: -13.5404816, -10.0399275, -13.5553703, -10.0384712, -2.6586676, 2.6725912
4: -15.5738468, -12.0072260, -15.5746527, -11.9973059, -2.6273069, 2.6172895
5: -6.1104980, -3.7488604, -6.1158009, -3.7464771, -1.6640058, 1.6670563
6: -3.6106141, -1.4266734, -3.6134858, -1.4037039, -1.8283033, 1.8073022
7: -7.6639872, -3.9012482, -7.6990137, -3.8997307, -3.4920263, 3.5275111
8: -2.8004570, -0.6195345, -2.8045912, -0.6177225, -1.9696569, 1.9723053
9: -9.3933372, -6.0077019, -9.3977757, -6.0057526, -2.4107170, 2.4118328

Time for backsubstitution: 14.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5734
type: B, layer: 1, pos: 5758
type: A, layer: 1, pos: 5758
type: B, layer: 1, pos: 4568
type: A, layer: 1, pos: 4568
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 4631
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 4631
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 467

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5734

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8100995, upper bound: 0.8100996
time: 4.57 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8100995, upper bound: 0.8100997
time: 6.97 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 8.8560371, 11.2455463, 8.8957262, 11.2440109, -1.9189248, 1.8713260
1: -19.3369961, -15.4389858, -19.2545948, -15.4456463, -3.2342458, 3.1465168
2: -3.8489280, -1.0337003, -3.8407452, -1.0405871, -2.2035260, 2.2034998
3: -13.5858612, -9.9870853, -13.5667381, -10.0373697, -2.7023053, 2.7246480
4: -15.6134892, -11.9788074, -15.5752602, -11.9897213, -2.6804476, 2.6456118
5: -6.1280670, -3.7288983, -6.1198387, -3.7446814, -1.6859608, 1.6916804
6: -3.6841943, -1.3764181, -3.6156561, -1.3861361, -1.8952980, 1.8493428
7: -7.7513666, -3.7920871, -7.7257991, -3.8985810, -3.5665483, 3.6269584
8: -2.8119488, -0.5976191, -2.8077497, -0.6163607, -1.9812498, 1.9965258
9: -9.4152441, -5.9926920, -9.4011021, -6.0042877, -2.4337835, 2.4317834

Time for backsubstitution: 14.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5758
type: B, layer: 1, pos: 5758
type: B, layer: 1, pos: 5734
type: B, layer: 1, pos: 4568
type: A, layer: 1, pos: 4568
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 4631
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 4631
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 467

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5758

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8162143, upper bound: 0.8193739
time: 5.09 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8193747, upper bound: 0.8193737
time: 4.88 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 24.43 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 24.43
Output dim: 0, lower bound: -0.8100995, upper bound: 0.8100996
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 24.43
Output dim: 0, lower bound: -0.8100995, upper bound: 0.8100997
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 24.43
Output dim: 0, lower bound: -0.8162143, upper bound: 0.8193739
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 24.43
Output dim: 0, lower bound: -0.8193747, upper bound: 0.8193737

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: 8.8822489, 11.2298317, 8.9059734, 11.2433739, -1.8904698, 1.8424313
1: -19.2956505, -15.4667501, -19.2501163, -15.4590816, -3.1506948, 3.0916905
2: -3.8283813, -1.0545433, -3.8342500, -1.0433496, -2.1826587, 2.1834743
3: -13.5479069, -10.0233917, -13.5544214, -10.0418453, -2.6000919, 2.6313159
4: -15.5949945, -11.9955511, -15.5736418, -11.9963751, -2.6586256, 2.6295867
5: -6.1168823, -3.7478588, -6.1183400, -3.7481871, -1.6682925, 1.6674833
6: -3.6684625, -1.3856263, -3.6113696, -1.3880699, -1.8857541, 1.8423636
7: -7.7090282, -3.8408487, -7.7211671, -3.9186878, -3.4884424, 3.5595963
8: -2.7906318, -0.6103935, -2.8014379, -0.6174631, -1.9600897, 1.9777179
9: -9.3914280, -6.0158410, -9.3917141, -6.0065665, -2.4236374, 2.4093983

Time for backsubstitution: 14.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5734
type: B, layer: 1, pos: 4568
type: A, layer: 1, pos: 4568
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 5758
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 4631
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 4631
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5734

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8069357, upper bound: 0.8191134
time: 5.11 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8069357, upper bound: 0.8193748
time: 5.24 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: 8.8560467, 11.2455463, 8.8957329, 11.2440128, -1.9035463, 1.8713224
1: -19.3369923, -15.4390011, -19.2545910, -15.4456520, -3.2137284, 3.1267252
2: -3.8489184, -1.0337025, -3.8407412, -1.0405898, -2.1971488, 2.2034926
3: -13.5858421, -9.9870930, -13.5667286, -10.0373707, -2.6862068, 2.7038231
4: -15.6134863, -11.9788151, -15.5752621, -11.9897232, -2.6784911, 2.6362667
5: -6.1280642, -3.7289014, -6.1198368, -3.7446833, -1.6852694, 1.6907833
6: -3.6841879, -1.3764200, -3.6156540, -1.3861362, -1.8899443, 1.8525350
7: -7.7513618, -3.7921023, -7.7257948, -3.8985898, -3.5665340, 3.5984106
8: -2.8119426, -0.5976200, -2.8077459, -0.6163626, -1.9746094, 1.9965196
9: -9.4152336, -5.9926934, -9.4010963, -6.0042887, -2.4139915, 2.4317741

Time for backsubstitution: 14.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5734
type: A, layer: 1, pos: 4568
type: B, layer: 1, pos: 4568
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 5758
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 4631
type: B, layer: 1, pos: 920
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 4631
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 467

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5734

## Relational analysis of IS_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8100974, upper bound: 0.8191137
time: 5.08 seconds

## Relational analysis of IS_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8100973, upper bound: 0.8193746
time: 6.27 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 25.81 seconds
IS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 25.81
Output dim: 0, lower bound: -0.8069357, upper bound: 0.8191134
IS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 25.81
Output dim: 0, lower bound: -0.8069357, upper bound: 0.8193748
IS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 25.81
Output dim: 0, lower bound: -0.8100974, upper bound: 0.8191137
IS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 25.81
Output dim: 0, lower bound: -0.8100973, upper bound: 0.8193746

## BFS IS instance: IS_A2_A1_B1

### Backsubstitution after applying IS history:
0: 8.8824577, 11.2298117, 8.9092064, 11.2234764, -1.8697546, 1.8492599
1: -19.2938023, -15.4667988, -19.2448959, -15.5022354, -3.1060820, 3.0986586
2: -3.8282371, -1.0546751, -3.8315091, -1.0506556, -2.1736431, 2.1781535
3: -13.5459995, -10.0234051, -13.5281382, -10.0439730, -2.5967159, 2.6031637
4: -15.5938187, -11.9956284, -15.5725193, -12.0139036, -2.6335206, 2.6262689
5: -6.1167693, -3.7479181, -6.1096773, -3.7523737, -1.6660137, 1.6583819
6: -3.6683784, -1.3860326, -3.6069772, -1.4286234, -1.8431239, 1.8482955
7: -7.7087379, -3.8417661, -7.6593332, -3.9212945, -3.4924998, 3.4927275
8: -2.7905474, -0.6113415, -2.7941399, -0.6204071, -1.9587569, 1.9685655
9: -9.3906355, -6.0159736, -9.3839703, -6.0099616, -2.4181499, 2.4002159

Time for backsubstitution: 14.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4568
type: A, layer: 1, pos: 4568
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 5758
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 4631
type: A, layer: 1, pos: 4631
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4568

## Relational analysis of IS_A2_A1_B1_B1

### Relational analysis result of IS_A2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8069326, upper bound: 0.8181442
time: 6.33 seconds

## Relational analysis of IS_A2_A1_B1_B2

### Relational analysis result of IS_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8069326, upper bound: 0.8191103
time: 4.71 seconds

## BFS IS instance: IS_A2_A1_B2

### Backsubstitution after applying IS history:
0: 8.8822489, 11.2298317, 8.8697824, 11.2449074, -1.8496060, 1.8457155
1: -19.2956505, -15.4667501, -19.3294640, -15.4524317, -3.1173019, 3.1351256
2: -3.8283813, -1.0545433, -3.8424397, -1.0380445, -2.1827393, 2.1856091
3: -13.5479069, -10.0233917, -13.5734911, -9.9964123, -2.6333313, 2.6437275
4: -15.5949945, -11.9955511, -15.6085596, -11.9854918, -2.6649070, 2.6670651
5: -6.1168823, -3.7478588, -6.1195078, -3.7323904, -1.6759677, 1.6590314
6: -3.6684625, -1.3856263, -3.6725702, -1.3783405, -1.8740773, 1.8683510
7: -7.7090282, -3.8408487, -7.7467885, -3.8128660, -3.5457287, 3.5661631
8: -2.7906318, -0.6103935, -2.8056364, -0.6012821, -1.9677005, 1.9725919
9: -9.3914280, -6.0158410, -9.4058094, -5.9951882, -2.4337668, 2.4214818

Time for backsubstitution: 14.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4568
type: A, layer: 1, pos: 4568
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 5758
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 4631
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 920
type: A, layer: 1, pos: 4631
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4568

## Relational analysis of IS_A2_A1_B2_B1

### Relational analysis result of IS_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8069326, upper bound: 0.8184021
time: 5.21 seconds

## Relational analysis of IS_A2_A1_B2_B2

### Relational analysis result of IS_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8069326, upper bound: 0.8191113
time: 7.00 seconds

## BFS IS instance: IS_A2_A2_B1

### Backsubstitution after applying IS history:
0: 8.8562574, 11.2455244, 8.8992825, 11.2241116, -1.8828321, 1.8778343
1: -19.3351479, -15.4390545, -19.2491131, -15.4887981, -3.1690750, 3.1333790
2: -3.8487780, -1.0338359, -3.8380022, -1.0480471, -2.1877174, 2.1980281
3: -13.5839357, -9.9871054, -13.5404739, -10.0399323, -2.6823244, 2.6760910
4: -15.6123085, -11.9788942, -15.5738459, -12.0072289, -2.6534872, 2.6325769
5: -6.1279526, -3.7289600, -6.1104965, -3.7488625, -1.6829975, 1.6806886
6: -3.6841030, -1.3768251, -3.6106122, -1.4266744, -1.8473244, 1.8578129
7: -7.7510672, -3.7930179, -7.6639814, -3.9012561, -3.5705261, 3.5314600
8: -2.8118544, -0.5985680, -2.8004551, -0.6195374, -1.9730477, 1.9873891
9: -9.4144440, -5.9928317, -9.3933315, -6.0077033, -2.4084854, 2.4227026

Time for backsubstitution: 14.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4568
type: B, layer: 1, pos: 4568
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 5758
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 4631
type: B, layer: 1, pos: 920
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 4631
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 467

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4568

## Relational analysis of IS_A2_A2_B1_A1

### Relational analysis result of IS_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8091191, upper bound: 0.8191105
time: 6.05 seconds

## Relational analysis of IS_A2_A2_B1_A2

### Relational analysis result of IS_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8100941, upper bound: 0.8191105
time: 4.90 seconds

## BFS IS instance: IS_A2_A2_B2

### Backsubstitution after applying IS history:
0: 8.8560467, 11.2455463, 8.8560410, 11.2455454, -1.8669014, 1.8777285
1: -19.3369923, -15.4390011, -19.3369942, -15.4389935, -3.1936893, 3.1739082
2: -3.8489184, -1.0337025, -3.8489246, -1.0336998, -2.1992126, 2.2055826
3: -13.5858421, -9.9870930, -13.5858526, -9.9870872, -2.7154226, 2.7183642
4: -15.6134863, -11.9788151, -15.6134863, -11.9788094, -2.6851940, 2.6778688
5: -6.1280642, -3.7289014, -6.1280651, -3.7288990, -1.6929379, 1.6927311
6: -3.6841879, -1.3764200, -3.6841898, -1.3764188, -1.8789229, 1.8856440
7: -7.7513618, -3.7921023, -7.7513647, -3.7920942, -3.6055880, 3.6050189
8: -2.8119426, -0.5976200, -2.8119450, -0.5976191, -1.9847856, 1.9914217
9: -9.4152336, -5.9926934, -9.4152412, -5.9926925, -2.4243693, 2.4441524

Time for backsubstitution: 14.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4568
type: B, layer: 1, pos: 4568
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 5758
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 4631
type: B, layer: 1, pos: 920
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 4631
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 467

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4568

## Relational analysis of IS_A2_A2_B2_A1

### Relational analysis result of IS_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8091191, upper bound: 0.8193738
time: 5.59 seconds

## Relational analysis of IS_A2_A2_B2_A2

### Relational analysis result of IS_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8100940, upper bound: 0.8193713
time: 5.79 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 26.14 seconds
IS_A2_A1_B1_B1, status: Status.VERIFIED, split count: 4, time: 26.14
Output dim: 0, lower bound: -0.8069326, upper bound: 0.8181442
IS_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 26.14
Output dim: 0, lower bound: -0.8069326, upper bound: 0.8191103
IS_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 26.14
Output dim: 0, lower bound: -0.8069326, upper bound: 0.8184021
IS_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 26.14
Output dim: 0, lower bound: -0.8069326, upper bound: 0.8191113
IS_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 26.14
Output dim: 0, lower bound: -0.8091191, upper bound: 0.8191105
IS_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 26.14
Output dim: 0, lower bound: -0.8100941, upper bound: 0.8191105
IS_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 26.14
Output dim: 0, lower bound: -0.8091191, upper bound: 0.8193738
IS_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 26.14
Output dim: 0, lower bound: -0.8100940, upper bound: 0.8193713

## BFS IS instance: IS_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: 8.8824635, 11.2298126, 8.9092178, 11.2234745, -1.8629539, 1.8457632
1: -19.2937984, -15.4668016, -19.2448826, -15.5022392, -3.0959125, 3.0940218
2: -3.8282349, -1.0546753, -3.8315010, -1.0506572, -2.1733170, 2.1786721
3: -13.5459909, -10.0234070, -13.5281181, -10.0439739, -2.5943308, 2.5909390
4: -15.5938168, -11.9956350, -15.5725174, -12.0139208, -2.6299381, 2.6261945
5: -6.1167665, -3.7479186, -6.1096725, -3.7523775, -1.6660082, 1.6553071
6: -3.6683686, -1.3860338, -3.6069603, -1.4286268, -1.8321738, 1.8318682
7: -7.7087345, -3.8417792, -7.6593266, -3.9213316, -3.4783807, 3.4798062
8: -2.7905440, -0.6113415, -2.7941337, -0.6204090, -1.9587517, 1.9679227
9: -9.3906317, -6.0159740, -9.3839645, -6.0099626, -2.4178391, 2.3998213

Time for backsubstitution: 14.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 5758
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 4631
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 4631
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 4568

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 820

## Relational analysis of IS_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 820

## Relational analysis of IS_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5758

## Relational analysis of IS_A2_A1_B1_B2_B1

### Relational analysis result of IS_A2_A1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8069326, upper bound: 0.8161829
time: 4.70 seconds

## Relational analysis of IS_A2_A1_B1_B2_B2

### Relational analysis result of IS_A2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8069326, upper bound: 0.8191101
time: 5.46 seconds

## BFS IS instance: IS_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: 8.8868809, 11.2296181, 8.8853245, 11.2371721, -1.8343740, 1.8292370
1: -19.2897091, -15.4677448, -19.3112469, -15.4744320, -3.0899439, 3.1147032
2: -3.8255136, -1.0562941, -3.8230262, -1.0477772, -2.1702251, 2.1600678
3: -13.5424156, -10.0242996, -13.5483751, -10.0164623, -2.6101847, 2.6074467
4: -15.5941038, -12.0011148, -15.5964375, -11.9997234, -2.6458750, 2.6494865
5: -6.1146421, -3.7491341, -6.1142025, -3.7477980, -1.6564875, 1.6524813
6: -3.6596980, -1.3873197, -3.6498120, -1.3965912, -1.8460145, 1.8444974
7: -7.7063446, -3.8514469, -7.7213717, -3.8379023, -3.5187235, 3.5260720
8: -2.7883425, -0.6111031, -2.7958260, -0.6087942, -1.9565492, 1.9617677
9: -9.3894634, -6.0169392, -9.3969030, -6.0050163, -2.4196434, 2.4093816

Time for backsubstitution: 14.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5758
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 4568
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 4631
type: B, layer: 1, pos: 920
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 4631
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5758

## Relational analysis of IS_A2_A1_B2_B1_B1

### Relational analysis result of IS_A2_A1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8069326, upper bound: 0.8154670
time: 4.75 seconds

## Relational analysis of IS_A2_A1_B2_B1_B2

### Relational analysis result of IS_A2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8069326, upper bound: 0.8184053
time: 5.32 seconds

## BFS IS instance: IS_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: 8.8822536, 11.2298307, 8.8697920, 11.2449055, -1.8495984, 1.8422163
1: -19.2956429, -15.4667482, -19.3294525, -15.4524336, -3.1172905, 3.1304455
2: -3.8283784, -1.0545448, -3.8424320, -1.0380471, -2.1824107, 2.1861274
3: -13.5478983, -10.0233927, -13.5734768, -9.9964151, -2.6220336, 2.6315041
4: -15.5949955, -11.9955540, -15.6085596, -11.9855080, -2.6613226, 2.6669898
5: -6.1168799, -3.7478580, -6.1195040, -3.7323947, -1.6759624, 1.6559565
6: -3.6684532, -1.3856277, -3.6725526, -1.3783445, -1.8721499, 1.8519228
7: -7.7090263, -3.8408651, -7.7467856, -3.8128994, -3.5279760, 3.5532627
8: -2.7906284, -0.6103935, -2.8056307, -0.6012821, -1.9676962, 1.9719868
9: -9.3914232, -6.0158420, -9.4058056, -5.9951878, -2.4334550, 2.4210849

Time for backsubstitution: 14.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 5758
type: B, layer: 1, pos: 4631
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 920
type: A, layer: 1, pos: 4631
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 4568

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 820

## Relational analysis of IS_A2_A1_B2_B2_B1

### Relational analysis result of IS_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8056035, upper bound: 0.8193635
time: 5.17 seconds

## Relational analysis of IS_A2_A1_B2_B2_B2

### Relational analysis result of IS_A2_A1_B2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8056052, upper bound: 0.8179037
time: 5.30 seconds

## BFS IS instance: IS_A2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 8.8734894, 11.2377872, 8.9038811, 11.2238960, -1.8647101, 1.8625779
1: -19.3152351, -15.4612160, -19.2436562, -15.4899235, -3.1469145, 3.1063519
2: -3.8298326, -1.0444124, -3.8346879, -1.0497833, -2.1622276, 2.1844401
3: -13.5595093, -10.0094681, -13.5341997, -10.0408058, -2.6487999, 2.6497092
4: -15.5987778, -11.9931650, -15.5730639, -12.0128002, -2.6339064, 2.6136422
5: -6.1194620, -3.7443473, -6.1085191, -3.7501175, -1.6720686, 1.6618688
6: -3.6581056, -1.3950247, -3.6021407, -1.4283719, -1.8198767, 1.8300233
7: -7.7254567, -3.8183963, -7.6611667, -3.9118323, -3.5342369, 3.5037296
8: -2.8019772, -0.6072073, -2.7981367, -0.6201658, -1.9622941, 1.9749804
9: -9.4056950, -6.0028567, -9.3911858, -6.0088534, -2.3962669, 2.4082382

Time for backsubstitution: 14.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 5758
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 4568
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 920
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 4631
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 4631
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 820

## Relational analysis of IS_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 820

## Relational analysis of IS_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5758

## Relational analysis of IS_A2_A2_B1_A1_B1

### Relational analysis result of IS_A2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8091189, upper bound: 0.8159525
time: 6.06 seconds

## Relational analysis of IS_A2_A2_B1_A1_B2

### Relational analysis result of IS_A2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8091192, upper bound: 0.8159525
time: 6.07 seconds

## BFS IS instance: IS_A2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 8.8562689, 11.2455254, 8.8992901, 11.2241116, -1.8761687, 1.8778262
1: -19.3351364, -15.4390564, -19.2491074, -15.4887981, -3.1584301, 3.1333680
2: -3.8487687, -1.0338372, -3.8379991, -1.0480486, -2.1877379, 2.1979566
3: -13.5839214, -9.9871063, -13.5404634, -10.0399332, -2.6795769, 2.6646078
4: -15.6123066, -11.9789095, -15.5738468, -12.0072365, -2.6527271, 2.6290460
5: -6.1279469, -3.7289639, -6.1104937, -3.7488637, -1.6799402, 1.6806834
6: -3.6840844, -1.3768291, -3.6106036, -1.4266758, -1.8288348, 1.8576490
7: -7.7510619, -3.7930541, -7.6639810, -3.9012716, -3.5652618, 3.5139852
8: -2.8118482, -0.5985699, -2.8004522, -0.6195374, -1.9723692, 1.9873834
9: -9.4144392, -5.9928341, -9.3933277, -6.0077028, -2.4081826, 2.4219799

Time for backsubstitution: 14.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 4631
type: B, layer: 1, pos: 920
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 4631
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 5758
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 4568

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 820

## Relational analysis of IS_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 820

## Relational analysis of IS_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6137

## Relational analysis of IS_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 844

## Relational analysis of IS_A2_A2_B1_A2_B1

### Relational analysis result of IS_A2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8094604, upper bound: 0.8150717
time: 5.52 seconds

## Relational analysis of IS_A2_A2_B1_A2_B2

### Relational analysis result of IS_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8094582, upper bound: 0.8184763
time: 5.22 seconds

## BFS IS instance: IS_A2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 8.8732777, 11.2378025, 8.8608122, 11.2453270, -1.8489022, 1.8623161
1: -19.3170776, -15.4611664, -19.3313694, -15.4401197, -3.1711378, 3.1466446
2: -3.8299782, -1.0442793, -3.8456149, -1.0355340, -2.1736040, 2.1918981
3: -13.5614176, -10.0094538, -13.5795698, -9.9882011, -2.6815910, 2.6919866
4: -15.5999546, -11.9930849, -15.6125021, -11.9843779, -2.6655941, 2.6587090
5: -6.1195726, -3.7442865, -6.1257000, -3.7301459, -1.6820121, 1.6733706
6: -3.6581907, -1.3946190, -3.6753535, -1.3781126, -1.8513823, 1.8574748
7: -7.7257485, -3.8174796, -7.7485600, -3.8026986, -3.5650291, 3.5775285
8: -2.8020620, -0.6062574, -2.8096352, -0.5983715, -1.9739037, 1.9790082
9: -9.4064846, -6.0027223, -9.4131031, -5.9938560, -2.4121342, 2.4296935

Time for backsubstitution: 14.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 5758
type: B, layer: 1, pos: 4568
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 4631
type: B, layer: 1, pos: 920
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 4631
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 467

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 820

## Relational analysis of IS_A2_A2_B2_A1_A1

### Relational analysis result of IS_A2_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8077899, upper bound: 0.8088003
time: 5.42 seconds

## Relational analysis of IS_A2_A2_B2_A1_A2

### Relational analysis result of IS_A2_A2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8077899, upper bound: 0.8179030
time: 6.72 seconds

## BFS IS instance: IS_A2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 8.8560581, 11.2455425, 8.8560448, 11.2455454, -1.8633218, 1.8777199
1: -19.3369808, -15.4390030, -19.3369904, -15.4389973, -3.1884594, 3.1738987
2: -3.8489089, -1.0337051, -3.8489203, -1.0337023, -2.1992369, 2.2055075
3: -13.5858278, -9.9870930, -13.5858450, -9.9870911, -2.7040253, 2.7068810
4: -15.6134853, -11.9788303, -15.6134882, -11.9788179, -2.6844282, 2.6743369
5: -6.1280584, -3.7289047, -6.1280632, -3.7289009, -1.6898813, 1.6927252
6: -3.6841702, -1.3764248, -3.6841824, -1.3764212, -1.8623700, 1.8795769
7: -7.7513547, -3.7921381, -7.7513618, -3.7921114, -3.5927153, 3.5875535
8: -2.8119359, -0.5976229, -2.8119440, -0.5976200, -1.9841089, 1.9914160
9: -9.4152279, -5.9926963, -9.4152374, -5.9926929, -2.4240656, 2.4413309

Time for backsubstitution: 14.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 4631
type: B, layer: 1, pos: 920
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 4631
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 5758
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 4568

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 820

## Relational analysis of IS_A2_A2_B2_A2_A1

### Relational analysis result of IS_A2_A2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8087662, upper bound: 0.8088002
time: 4.85 seconds

## Relational analysis of IS_A2_A2_B2_A2_A2

### Relational analysis result of IS_A2_A2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8087662, upper bound: 0.8179050
time: 9.13 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 28.62 seconds
IS_A2_A1_B1_B2_B1, status: Status.VERIFIED, split count: 5, time: 28.62
Output dim: 0, lower bound: -0.8069326, upper bound: 0.8161829
IS_A2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 28.62
Output dim: 0, lower bound: -0.8069326, upper bound: 0.8191101
IS_A2_A1_B2_B1_B1, status: Status.VERIFIED, split count: 5, time: 28.62
Output dim: 0, lower bound: -0.8069326, upper bound: 0.8154670
IS_A2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 28.62
Output dim: 0, lower bound: -0.8069326, upper bound: 0.8184053
IS_A2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 28.62
Output dim: 0, lower bound: -0.8056035, upper bound: 0.8193635
IS_A2_A1_B2_B2_B2, status: Status.VERIFIED, split count: 5, time: 28.62
Output dim: 0, lower bound: -0.8056052, upper bound: 0.8179037
IS_A2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 5, time: 28.62
Output dim: 0, lower bound: -0.8091189, upper bound: 0.8159525
IS_A2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 5, time: 28.62
Output dim: 0, lower bound: -0.8091192, upper bound: 0.8159525
IS_A2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 5, time: 28.62
Output dim: 0, lower bound: -0.8094604, upper bound: 0.8150717
IS_A2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 28.62
Output dim: 0, lower bound: -0.8094582, upper bound: 0.8184763
IS_A2_A2_B2_A1_A1, status: Status.VERIFIED, split count: 5, time: 28.62
Output dim: 0, lower bound: -0.8077899, upper bound: 0.8088003
IS_A2_A2_B2_A1_A2, status: Status.VERIFIED, split count: 5, time: 28.62
Output dim: 0, lower bound: -0.8077899, upper bound: 0.8179030
IS_A2_A2_B2_A2_A1, status: Status.VERIFIED, split count: 5, time: 28.62
Output dim: 0, lower bound: -0.8087662, upper bound: 0.8088002
IS_A2_A2_B2_A2_A2, status: Status.VERIFIED, split count: 5, time: 28.62
Output dim: 0, lower bound: -0.8087662, upper bound: 0.8179050

## BFS IS instance: IS_A2_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: 8.8824635, 11.2298126, 8.8998184, 11.2240820, -1.8506091, 1.8548942
1: -19.2937984, -15.4668016, -19.2487965, -15.4905863, -3.0991983, 3.0971336
2: -3.8282349, -1.0546753, -3.8376832, -1.0489969, -2.1741228, 2.1792355
3: -13.5459909, -10.0234070, -13.5403118, -10.0419321, -2.5879302, 2.5896049
4: -15.5938168, -11.9956350, -15.5735531, -12.0073099, -2.6339989, 2.6242404
5: -6.1167665, -3.7479186, -6.1103296, -3.7489448, -1.6676016, 1.6559711
6: -3.6683686, -1.3860338, -3.6103730, -1.4267383, -1.8289058, 1.8324163
7: -7.7087345, -3.8417792, -7.6623840, -3.9013650, -3.4996004, 3.4541659
8: -2.7905440, -0.6113415, -2.8002377, -0.6197634, -1.9594584, 1.9739385
9: -9.3906317, -6.0159740, -9.3933029, -6.0086861, -2.4181452, 2.4105394

Time for backsubstitution: 14.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 4631
type: B, layer: 1, pos: 920
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 4568
type: A, layer: 1, pos: 4631
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 820

## Relational analysis of IS_A2_A1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 820

## Relational analysis of IS_A2_A1_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 844

## Relational analysis of IS_A2_A1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 844

## Relational analysis of IS_A2_A1_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6137

## Relational analysis of IS_A2_A1_B1_B2_B2_A1

### Relational analysis result of IS_A2_A1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8062072, upper bound: 0.8182912
time: 5.27 seconds

## Relational analysis of IS_A2_A1_B1_B2_B2_A2

### Relational analysis result of IS_A2_A1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8069325, upper bound: 0.8191102
time: 6.39 seconds

## BFS IS instance: IS_A2_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: 8.8868809, 11.2296181, 8.8760338, 11.2377825, -1.8350787, 1.8385963
1: -19.2897091, -15.4677448, -19.3148632, -15.4627819, -3.1015244, 3.1178708
2: -3.8255136, -1.0562941, -3.8297114, -1.0461140, -2.1717510, 2.1616530
3: -13.5424156, -10.0242996, -13.5612984, -10.0144205, -2.5913386, 2.6069417
4: -15.5941038, -12.0011148, -15.5974874, -11.9931355, -2.6507425, 2.6475115
5: -6.1146421, -3.7491341, -6.1148596, -3.7443528, -1.6581426, 1.6531348
6: -3.6596980, -1.3873197, -3.6531541, -1.3946695, -1.8434377, 1.8449562
7: -7.7063446, -3.8514469, -7.7243795, -3.8179703, -3.5228896, 3.5006537
8: -2.7883425, -0.6111031, -2.8018866, -0.6081553, -1.9572525, 1.9678516
9: -9.3894634, -6.0169392, -9.4064713, -6.0037088, -2.4198947, 2.4208343

Time for backsubstitution: 14.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 4631
type: B, layer: 1, pos: 920
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 4631
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 4568
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 820

## Relational analysis of IS_A2_A1_B2_B1_B2_A1

### Relational analysis result of IS_A2_A1_B2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8056052, upper bound: 0.8078245
time: 5.31 seconds

## Relational analysis of IS_A2_A1_B2_B1_B2_A2

### Relational analysis result of IS_A2_A1_B2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8056052, upper bound: 0.8169367
time: 4.93 seconds

## BFS IS instance: IS_A2_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: 8.8822536, 11.2298307, 8.8701096, 11.2427521, -1.8475003, 1.8415244
1: -19.2956429, -15.4667482, -19.3288212, -15.4594650, -3.1102085, 3.1299448
2: -3.8283784, -1.0545448, -3.8422327, -1.0386031, -2.1818967, 2.1854129
3: -13.5478983, -10.0233927, -13.5701323, -9.9966507, -2.6217227, 2.6279531
4: -15.5949955, -11.9955540, -15.6084490, -11.9895391, -2.6570544, 2.6668200
5: -6.1168799, -3.7478580, -6.1191859, -3.7327788, -1.6755610, 1.6555605
6: -3.6684532, -1.3856277, -3.6721132, -1.3835838, -1.8666730, 1.8514850
7: -7.7090263, -3.8408651, -7.7382112, -3.8130341, -3.5277872, 3.5441782
8: -2.7906284, -0.6103935, -2.8047643, -0.6015806, -1.9673414, 1.9709921
9: -9.3914232, -6.0158420, -9.4048948, -5.9954576, -2.4331312, 2.4203427

Time for backsubstitution: 14.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 5758
type: B, layer: 1, pos: 4631
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 920
type: A, layer: 1, pos: 4631
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 4568

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 820

## Relational analysis of IS_A2_A1_B2_B2_B1_A1

### Relational analysis result of IS_A2_A1_B2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8056030, upper bound: 0.8088009
time: 4.91 seconds

## Relational analysis of IS_A2_A1_B2_B2_B1_A2

### Relational analysis result of IS_A2_A1_B2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8056030, upper bound: 0.8179040
time: 5.08 seconds

## BFS IS instance: IS_A2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 8.8562717, 11.2455244, 8.8972149, 11.2299118, -1.8775713, 1.8794003
1: -19.3332443, -15.4390574, -19.2518101, -15.4909744, -3.1600785, 3.1409650
2: -3.8487582, -1.0338403, -3.8419559, -1.0376363, -2.1943364, 2.2042658
3: -13.5839062, -9.9871082, -13.5471611, -10.0273266, -2.6859384, 2.6727648
4: -15.6123085, -11.9789171, -15.5816326, -12.0047140, -2.6518621, 2.6296902
5: -6.1255608, -3.7289653, -6.1087904, -3.7385514, -1.6887145, 1.6836526
6: -3.6837115, -1.3768315, -3.6124032, -1.4220567, -1.8312798, 1.8584890
7: -7.7510557, -3.7930684, -7.6798325, -3.8983858, -3.5673079, 3.5180326
8: -2.8118443, -0.5985699, -2.8038664, -0.6146708, -1.9788175, 1.9892659
9: -9.4144344, -5.9928341, -9.3949432, -5.9984627, -2.4163842, 2.4236522

Time for backsubstitution: 14.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 4631
type: B, layer: 1, pos: 920
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 4631
type: B, layer: 1, pos: 5758
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 4568

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 820

## Relational analysis of IS_A2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 820

## Relational analysis of IS_A2_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6137

## Relational analysis of IS_A2_A2_B1_A2_B2_B1

### Relational analysis result of IS_A2_A2_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8085404, upper bound: 0.8177942
time: 4.99 seconds

## Relational analysis of IS_A2_A2_B1_A2_B2_B2

### Relational analysis result of IS_A2_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8094581, upper bound: 0.8184763
time: 5.04 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 36.11 seconds
IS_A2_A1_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 36.11
Output dim: 0, lower bound: -0.8062072, upper bound: 0.8182912
IS_A2_A1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 36.11
Output dim: 0, lower bound: -0.8069325, upper bound: 0.8191102
IS_A2_A1_B2_B1_B2_A1, status: Status.VERIFIED, split count: 6, time: 36.11
Output dim: 0, lower bound: -0.8056052, upper bound: 0.8078245
IS_A2_A1_B2_B1_B2_A2, status: Status.VERIFIED, split count: 6, time: 36.11
Output dim: 0, lower bound: -0.8056052, upper bound: 0.8169367
IS_A2_A1_B2_B2_B1_A1, status: Status.VERIFIED, split count: 6, time: 36.11
Output dim: 0, lower bound: -0.8056030, upper bound: 0.8088009
IS_A2_A1_B2_B2_B1_A2, status: Status.VERIFIED, split count: 6, time: 36.11
Output dim: 0, lower bound: -0.8056030, upper bound: 0.8179040
IS_A2_A2_B1_A2_B2_B1, status: Status.VERIFIED, split count: 6, time: 36.11
Output dim: 0, lower bound: -0.8085404, upper bound: 0.8177942
IS_A2_A2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 36.11
Output dim: 0, lower bound: -0.8094581, upper bound: 0.8184763

## BFS IS instance: IS_A2_A1_B1_B2_B2_A1

### Backsubstitution after applying IS history:
0: 8.8902121, 11.2211275, 8.9007826, 11.2225723, -1.8401337, 1.8452935
1: -19.2802181, -15.4771633, -19.2466755, -15.4919243, -3.0811920, 3.0834298
2: -3.8152044, -1.0604899, -3.8358188, -1.0497590, -2.1561246, 2.1684670
3: -13.5157490, -10.0616217, -13.5356674, -10.0481911, -2.5259686, 2.5454574
4: -15.5078640, -12.0142298, -15.5590115, -12.0098124, -2.5445361, 2.5682397
5: -6.1119347, -3.7729073, -6.1099997, -3.7532110, -1.6570091, 1.6302102
6: -3.6574574, -1.4020600, -3.6090670, -1.4291863, -1.8150392, 1.8156958
7: -7.6836138, -3.8872337, -7.6587391, -3.9090474, -3.4527893, 3.4065816
8: -2.7788544, -0.6261263, -2.7985840, -0.6221600, -1.9417171, 1.9585218
9: -9.3772326, -6.0544558, -9.3915548, -6.0152345, -2.3848257, 2.3685851

Time for backsubstitution: 14.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 920
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 4631
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 4631
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 4568

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 820

## Relational analysis of IS_A2_A1_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 820

## Relational analysis of IS_A2_A1_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 844

## Relational analysis of IS_A2_A1_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 844

## Relational analysis of IS_A2_A1_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 920

## Relational analysis of IS_A2_A1_B1_B2_B2_A1_B1

### Relational analysis result of IS_A2_A1_B1_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8062054, upper bound: 0.8169296
time: 5.80 seconds

## Relational analysis of IS_A2_A1_B1_B2_B2_A1_B2

### Relational analysis result of IS_A2_A1_B1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8062054, upper bound: 0.8182899
time: 5.07 seconds

## BFS IS instance: IS_A2_A1_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: 8.8824615, 11.2298107, 8.8998184, 11.2240820, -1.8500321, 1.8611960
1: -19.2937965, -15.4668007, -19.2487965, -15.4905863, -3.0988703, 3.0999198
2: -3.8282351, -1.0546757, -3.8376832, -1.0489969, -2.1734614, 2.1859639
3: -13.5459909, -10.0234089, -13.5403118, -10.0419321, -2.5868788, 2.5793638
4: -15.5938148, -11.9956369, -15.5735531, -12.0073099, -2.5682001, 2.6242127
5: -6.1167669, -3.7479215, -6.1103296, -3.7489448, -1.6676011, 1.6542466
6: -3.6683683, -1.3860338, -3.6103730, -1.4267383, -1.8281415, 1.8393717
7: -7.7087326, -3.8417821, -7.6623840, -3.9013650, -3.5212345, 3.4541659
8: -2.7905436, -0.6113434, -2.8002377, -0.6197634, -1.9592323, 1.9773111
9: -9.3906326, -6.0159779, -9.3933029, -6.0086861, -2.4181442, 2.3932326

Time for backsubstitution: 14.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 4631
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 920
type: A, layer: 1, pos: 4631
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 4568
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 6137

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 820

## Relational analysis of IS_A2_A1_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 820

## Relational analysis of IS_A2_A1_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 844

## Relational analysis of IS_A2_A1_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 844

## Relational analysis of IS_A2_A1_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4631

## Relational analysis of IS_A2_A1_B1_B2_B2_A2_B1

### Relational analysis result of IS_A2_A1_B1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8063761, upper bound: 0.8191084
time: 5.19 seconds

## Relational analysis of IS_A2_A1_B1_B2_B2_A2_B2

### Relational analysis result of IS_A2_A1_B1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8069308, upper bound: 0.8191081
time: 5.15 seconds

## BFS IS instance: IS_A2_A2_B1_A2_B2_B2

### Backsubstitution after applying IS history:
0: 8.8562717, 11.2455244, 8.8972158, 11.2299109, -1.8827767, 1.8791432
1: -19.3332443, -15.4390574, -19.2518101, -15.4909725, -3.1652374, 3.1407385
2: -3.8487582, -1.0338403, -3.8419549, -1.0376354, -2.2019725, 2.2039547
3: -13.5839062, -9.9871082, -13.5471573, -10.0273285, -2.6724200, 2.6717234
4: -15.6123085, -11.9789171, -15.5816288, -12.0047150, -2.6518154, 2.5651922
5: -6.1255608, -3.7289653, -6.1087885, -3.7385528, -1.6869328, 1.6836517
6: -3.6837115, -1.3768315, -3.6124022, -1.4220574, -1.8370829, 1.8582883
7: -7.7510557, -3.7930684, -7.6798296, -3.8983893, -3.5673070, 3.5387344
8: -2.8118443, -0.5985699, -2.8038654, -0.6146736, -1.9824018, 1.9890113
9: -9.4144344, -5.9928341, -9.3949423, -5.9984641, -2.3979936, 2.4221730

Time for backsubstitution: 14.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 4631
type: B, layer: 1, pos: 920
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 4631
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 5758
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 4568
type: A, layer: 1, pos: 6137

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 820

## Relational analysis of IS_A2_A2_B1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 820

## Relational analysis of IS_A2_A2_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 844

## Relational analysis of IS_A2_A2_B1_A2_B2_B2_A1

### Relational analysis result of IS_A2_A2_B1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8060478, upper bound: 0.8184763
time: 12.73 seconds

## Relational analysis of IS_A2_A2_B1_A2_B2_B2_A2

### Relational analysis result of IS_A2_A2_B1_A2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8060478, upper bound: 0.8150714
time: 8.14 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 47.27 seconds
IS_A2_A1_B1_B2_B2_A1_B1, status: Status.VERIFIED, split count: 7, time: 47.27
Output dim: 0, lower bound: -0.8062054, upper bound: 0.8169296
IS_A2_A1_B1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 47.27
Output dim: 0, lower bound: -0.8062054, upper bound: 0.8182899
IS_A2_A1_B1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 47.27
Output dim: 0, lower bound: -0.8063761, upper bound: 0.8191084
IS_A2_A1_B1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 47.27
Output dim: 0, lower bound: -0.8069308, upper bound: 0.8191081
IS_A2_A2_B1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 47.27
Output dim: 0, lower bound: -0.8060478, upper bound: 0.8184763
IS_A2_A2_B1_A2_B2_B2_A2, status: Status.VERIFIED, split count: 7, time: 47.27
Output dim: 0, lower bound: -0.8060478, upper bound: 0.8150714

## BFS IS instance: IS_A2_A1_B1_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 8.8902121, 11.2211275, 8.9007826, 11.2225733, -1.8398604, 1.8433423
1: -19.2802181, -15.4771633, -19.2466698, -15.4919300, -3.0804110, 3.0806270
2: -3.8152044, -1.0604899, -3.8354950, -1.0497612, -2.1558595, 2.1659994
3: -13.5157490, -10.0616217, -13.5356636, -10.0481901, -2.5259681, 2.5453222
4: -15.5078640, -12.0142298, -15.5590096, -12.0098495, -2.5470009, 2.5682397
5: -6.1119347, -3.7729073, -6.1099987, -3.7532115, -1.6543782, 1.6302090
6: -3.6574574, -1.4020600, -3.6090651, -1.4291873, -1.8149595, 1.8158131
7: -7.6836138, -3.8872337, -7.6587343, -3.9090476, -3.4530578, 3.4065773
8: -2.7788544, -0.6261263, -2.7985787, -0.6221609, -1.9417143, 1.9626732
9: -9.3772326, -6.0544558, -9.3914957, -6.0152373, -2.3865199, 2.3685839

Time for backsubstitution: 14.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 4631
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 4631
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 4568

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 820

## Relational analysis of IS_A2_A1_B1_B2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 820

## Relational analysis of IS_A2_A1_B1_B2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 844

## Relational analysis of IS_A2_A1_B1_B2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 2): status=Status.UNKNOWN, k_low=4, k_high=4, k_mid=4, eps_mid=0.0156250, abs_max=1.8807220458984375
rel_dist={0: [-0.8193859179682992, 0.8193858024795073]}

## Binary Search with IS_dual Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01171875
execution time: 1654.54 seconds
