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
execution time: IAR + LP analysis = 14.92 + 32.53 = 47.45 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3552.55 seconds, max iter: 100)

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
Binary search time: 155.20 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.01171875


# Individual Split (IS_dual_ind) starts
Time budget: 3397.35 seconds

## Binary search (step 0) starts
Candidate k: 8, corresponding eps: 0.0312500


## IAR start
Binary search (step 0): status=Status.ADV_EXAMPLE, k_low=4, k_high=12, k_mid=8, eps_mid=0.0312500, abs_max=None

## Binary search (step 1) starts
Candidate k: 5, corresponding eps: 0.0195312


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5734
type: A, layer: 1, pos: 5758
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 4568
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 4631
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 52

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5734

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9368811, upper bound: 0.9255753
time: 7.45 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9371562, upper bound: 0.9371556
time: 6.73 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 14.34 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 14.34
Output dim: 0, lower bound: -0.9368811, upper bound: 0.9255753
IS_A2, status: Status.UNKNOWN, split count: 1, time: 14.34
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

Time for backsubstitution: 14.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5734
type: B, layer: 1, pos: 5758
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 4568
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 4631
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 52

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5734

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9255730, upper bound: 0.9255753
time: 6.29 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9255730, upper bound: 0.9255733
time: 5.43 seconds

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

Time for backsubstitution: 14.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5734
type: B, layer: 1, pos: 5758
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 4568
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 4631
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 52

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5734

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9255730, upper bound: 0.9368811
time: 5.57 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9255730, upper bound: 0.9371558
time: 4.59 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 24.61 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 24.61
Output dim: 0, lower bound: -0.9255730, upper bound: 0.9255753
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 24.61
Output dim: 0, lower bound: -0.9255730, upper bound: 0.9255733
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 24.61
Output dim: 0, lower bound: -0.9255730, upper bound: 0.9368811
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 24.61
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

Time for backsubstitution: 14.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5758
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 4568
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 4631
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 52

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5758

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9216506, upper bound: 0.9255716
time: 4.69 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9255725, upper bound: 0.9255707
time: 7.57 seconds

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

Time for backsubstitution: 14.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5758
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 4568
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 4631
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 52

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5758

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9216506, upper bound: 0.9255699
time: 4.68 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9255727, upper bound: 0.9255705
time: 4.55 seconds

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

Time for backsubstitution: 14.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5758
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 4568
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 4631
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 52

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5758

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9216482, upper bound: 0.9368776
time: 5.06 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9255703, upper bound: 0.9368778
time: 6.30 seconds

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

Time for backsubstitution: 14.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5758
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 4568
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 4631
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 52

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5758

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9216482, upper bound: 0.9371560
time: 5.42 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9255703, upper bound: 0.9371541
time: 6.10 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 26.20 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 26.20
Output dim: 0, lower bound: -0.9216506, upper bound: 0.9255716
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 26.20
Output dim: 0, lower bound: -0.9255725, upper bound: 0.9255707
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 26.20
Output dim: 0, lower bound: -0.9216506, upper bound: 0.9255699
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 26.20
Output dim: 0, lower bound: -0.9255727, upper bound: 0.9255705
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 26.20
Output dim: 0, lower bound: -0.9216482, upper bound: 0.9368776
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 26.20
Output dim: 0, lower bound: -0.9255703, upper bound: 0.9368778
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 26.20
Output dim: 0, lower bound: -0.9216482, upper bound: 0.9371560
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 26.20
Output dim: 0, lower bound: -0.9255703, upper bound: 0.9371541

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

Time for backsubstitution: 14.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5758
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 4568
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 4631
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 52

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5758

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9216505, upper bound: 0.9216528
time: 5.66 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9216505, upper bound: 0.9255728
time: 4.99 seconds

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

Time for backsubstitution: 14.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5758
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 4568
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 4631
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 52

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5758

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9255727, upper bound: 0.9216515
time: 4.73 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9255728, upper bound: 0.9255728
time: 4.36 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: 8.9226303, 11.2084036, 8.8651161, 11.2450047, -1.9274120, 1.9475095
1: -19.2102623, -15.5165377, -19.3323135, -15.4503460, -3.2226009, 3.2579041
2: -3.8174486, -1.0676342, -3.8434236, -1.0362502, -2.2491536, 2.2563794
3: -13.5026731, -10.0723238, -13.5748653, -9.9914150, -2.7602892, 2.7106290
4: -15.5580549, -12.0238962, -15.6113720, -11.9844446, -2.7386532, 2.7498326
5: -6.1050787, -3.7678983, -6.1260290, -3.7318377, -1.7583241, 1.7433801
6: -3.6007564, -1.4359775, -3.6797850, -1.3781562, -1.9157076, 1.8999298
7: -7.6213474, -3.9494665, -7.7473907, -3.8093374, -3.6101384, 3.6517467
8: -2.7791033, -0.6302538, -2.8066158, -0.5991201, -2.0272422, 2.0258989
9: -9.3696842, -6.0306206, -9.4070759, -5.9947019, -2.5012541, 2.5120182

Time for backsubstitution: 14.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5758
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 4568
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 4631
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 52

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5758

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9329543, upper bound: 0.9216490
time: 5.15 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9329543, upper bound: 0.9255727
time: 5.10 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: 8.8992863, 11.2241116, 8.8561077, 11.2455387, -1.9427238, 1.9627187
1: -19.2491074, -15.4888048, -19.3364105, -15.4390087, -3.2744865, 3.2912998
2: -3.8379967, -1.0480481, -3.8488798, -1.0337430, -2.2722802, 2.2738674
3: -13.5404644, -10.0399332, -13.5852489, -9.9870930, -2.7865534, 2.8066978
4: -15.5738430, -12.0072346, -15.6131163, -11.9788351, -2.7553263, 2.7637882
5: -6.1104951, -3.7488642, -6.1280308, -3.7289190, -1.7657616, 1.7677786
6: -3.6106095, -1.4266739, -3.6841643, -1.3765473, -1.9147596, 1.9160593
7: -7.6639819, -3.9012637, -7.7512712, -3.7923825, -3.6664782, 3.6857448
8: -2.8004494, -0.6195374, -2.8119192, -0.5979185, -2.0435143, 2.0416231
9: -9.3933258, -6.0077038, -9.4149904, -5.9927368, -2.5100117, 2.5340939

Time for backsubstitution: 14.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5758
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 4568
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 4631
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 52

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5758

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9368776, upper bound: 0.9216483
time: 9.15 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9368776, upper bound: 0.9255706
time: 18.92 seconds

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

Time for backsubstitution: 14.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5758
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 4568
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 4631
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 52

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5758

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9216482, upper bound: 0.9329565
time: 7.28 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9216482, upper bound: 0.9368798
time: 4.92 seconds

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

Time for backsubstitution: 14.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5758
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 4568
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 4631
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 52

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5758

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9255704, upper bound: 0.9329556
time: 4.68 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9255704, upper bound: 0.9368776
time: 4.95 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: 8.8822489, 11.2298317, 8.8650475, 11.2450094, -1.9268403, 1.9291036
1: -19.2956505, -15.4667501, -19.3328991, -15.4503288, -3.2624111, 3.2619700
2: -3.8283813, -1.0545433, -3.8434677, -1.0362083, -2.2570925, 2.2663991
3: -13.5479069, -10.0233917, -13.5754700, -9.9914103, -2.8025403, 2.7561150
4: -15.5949945, -11.9955511, -15.6117458, -11.9844160, -2.7806034, 2.7804189
5: -6.1168823, -3.7478588, -6.1260662, -3.7318182, -1.7621853, 1.7536728
6: -3.6684625, -1.3856263, -3.6798124, -1.3780277, -1.9396629, 1.9305327
7: -7.7090282, -3.8408487, -7.7474837, -3.8090475, -3.6846957, 3.7186513
8: -2.7906318, -0.6103935, -2.8066440, -0.5988188, -2.0312233, 2.0359578
9: -9.3914280, -6.0158410, -9.4073277, -5.9946589, -2.5225387, 2.5273132

Time for backsubstitution: 14.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5758
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 4568
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 4631
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 52

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5758

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9218558, upper bound: 0.9332316
time: 4.70 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9218558, upper bound: 0.9371537
time: 4.75 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: 8.8560467, 11.2455463, 8.8560410, 11.2455463, -1.9447932, 1.9550807
1: -19.3369923, -15.4390011, -19.3369942, -15.4389935, -3.3173389, 3.2985415
2: -3.8489184, -1.0337025, -3.8489246, -1.0337021, -2.2802649, 2.2857718
3: -13.5858421, -9.9870930, -13.5858517, -9.9870892, -2.8289304, 2.8298340
4: -15.6134863, -11.9788151, -15.6134882, -11.9788074, -2.8005190, 2.7945032
5: -6.1280642, -3.7289014, -6.1280661, -3.7288990, -1.7781978, 1.7779896
6: -3.6841879, -1.3764200, -3.6841915, -1.3764193, -1.9445167, 1.9512401
7: -7.7513618, -3.7921023, -7.7513647, -3.7920938, -3.7411880, 3.7433438
8: -2.8119426, -0.5976200, -2.8119469, -0.5976200, -2.0474596, 2.0537643
9: -9.4152336, -5.9926934, -9.4152403, -5.9926925, -2.5312343, 2.5497065

Time for backsubstitution: 14.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5758
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 4568
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 4631
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 52

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5758

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9257780, upper bound: 0.9332317
time: 4.85 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9257780, upper bound: 0.9371525
time: 5.02 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 24.44 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.44
Output dim: 0, lower bound: -0.9216505, upper bound: 0.9216528
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.44
Output dim: 0, lower bound: -0.9216505, upper bound: 0.9255728
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.44
Output dim: 0, lower bound: -0.9255727, upper bound: 0.9216515
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.44
Output dim: 0, lower bound: -0.9255728, upper bound: 0.9255728
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.44
Output dim: 0, lower bound: -0.9329543, upper bound: 0.9216490
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.44
Output dim: 0, lower bound: -0.9329543, upper bound: 0.9255727
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.44
Output dim: 0, lower bound: -0.9368776, upper bound: 0.9216483
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.44
Output dim: 0, lower bound: -0.9368776, upper bound: 0.9255706
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.44
Output dim: 0, lower bound: -0.9216482, upper bound: 0.9329565
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.44
Output dim: 0, lower bound: -0.9216482, upper bound: 0.9368798
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.44
Output dim: 0, lower bound: -0.9255704, upper bound: 0.9329556
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.44
Output dim: 0, lower bound: -0.9255704, upper bound: 0.9368776
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.44
Output dim: 0, lower bound: -0.9218558, upper bound: 0.9332316
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.44
Output dim: 0, lower bound: -0.9218558, upper bound: 0.9371537
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.44
Output dim: 0, lower bound: -0.9257780, upper bound: 0.9332317
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.44
Output dim: 0, lower bound: -0.9257780, upper bound: 0.9371525

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

Time for backsubstitution: 14.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 4568
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 4631
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 52

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 820

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4568

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9204908, upper bound: 0.9218099
time: 5.91 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9216469, upper bound: 0.9218097
time: 7.04 seconds

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

Time for backsubstitution: 14.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 4568
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 4631
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 52

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 820

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4568

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9204909, upper bound: 0.9255691
time: 4.81 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9216470, upper bound: 0.9255686
time: 4.98 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 8.8992863, 11.2241116, 8.9226303, 11.2084036, -1.9141541, 1.9063094
1: -19.2491074, -15.4888048, -19.2102623, -15.5165377, -3.1753988, 3.1855016
2: -3.8379967, -1.0480481, -3.8174486, -1.0676342, -2.2456703, 2.2372732
3: -13.5404644, -10.0399332, -13.5026731, -10.0723238, -2.6826887, 2.7199807
4: -15.5738430, -12.0072346, -15.5580549, -12.0238962, -2.7094245, 2.7161355
5: -6.1104951, -3.7488642, -6.1050787, -3.7678983, -1.7236085, 1.7390840
6: -3.6106095, -1.4266739, -3.6007564, -1.4359775, -1.8528066, 1.8622746
7: -7.6639819, -3.9012637, -7.6213474, -3.9494665, -3.5740423, 3.5634227
8: -2.8004494, -0.6195374, -2.7791033, -0.6302538, -2.0190239, 2.0082273
9: -9.3933258, -6.0077038, -9.3696842, -6.0306206, -2.5018206, 2.4886575

Time for backsubstitution: 14.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 4568
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 4631
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 52

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 820

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4568

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9244122, upper bound: 0.9216493
time: 4.86 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9255686, upper bound: 0.9216469
time: 7.60 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 8.8992863, 11.2241116, 8.8992863, 11.2241116, -1.9210052, 1.9210055
1: -19.2491074, -15.4888048, -19.2491074, -15.4888048, -3.2074242, 3.2074242
2: -3.8379967, -1.0480481, -3.8379967, -1.0480481, -2.2572289, 2.2572293
3: -13.5404644, -10.0399332, -13.5404644, -10.0399332, -2.7489758, 2.7489762
4: -15.5738430, -12.0072346, -15.5738430, -12.0072346, -2.7226067, 2.7226067
5: -6.1104951, -3.7488642, -6.1104951, -3.7488642, -1.7452655, 1.7452655
6: -3.6106095, -1.4266739, -3.6106095, -1.4266739, -1.8708334, 1.8708336
7: -7.6639819, -3.9012637, -7.6639819, -3.9012637, -3.6050053, 3.6050057
8: -2.8004494, -0.6195374, -2.8004494, -0.6195374, -2.0231571, 2.0231566
9: -9.3933258, -6.0077038, -9.3933258, -6.0077038, -2.4951062, 2.4951060

Time for backsubstitution: 14.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 4568
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 4631
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 52

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 820

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4568

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9244125, upper bound: 0.9216493
time: 4.80 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9255689, upper bound: 0.9216472
time: 7.18 seconds

## BFS IS instance: IS_A1_B2_A1_B1

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

Time for backsubstitution: 14.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 4568
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 4631
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 52

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 820

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4568

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9317947, upper bound: 0.9218074
time: 4.70 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9329516, upper bound: 0.9218074
time: 4.91 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 8.9226303, 11.2084036, 8.8561115, 11.2455378, -1.9280267, 1.9525683
1: -19.2102623, -15.5165377, -19.3364067, -15.4390163, -3.2337637, 3.2452338
2: -3.8174486, -1.0676342, -3.8488741, -1.0337453, -2.2523212, 2.2568088
3: -13.5026731, -10.0723238, -13.5852375, -9.9870930, -2.7431283, 2.7236366
4: -15.5580549, -12.0238962, -15.6131134, -11.9788380, -2.7427959, 2.7504830
5: -6.1050787, -3.7678983, -6.1280298, -3.7289200, -1.7593713, 1.7460785
6: -3.6007564, -1.4359775, -3.6841607, -1.3765495, -1.9129496, 1.9015383
7: -7.6213474, -3.9494665, -7.7512708, -3.7923930, -3.6139660, 3.6547780
8: -2.7791033, -0.6302538, -2.8119140, -0.5979204, -2.0285835, 2.0311904
9: -9.3696842, -6.0306206, -9.4149847, -5.9927363, -2.5035615, 2.5226712

Time for backsubstitution: 14.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 4568
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 4631
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 52

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 820

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4568

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9317947, upper bound: 0.9255671
time: 7.88 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9329507, upper bound: 0.9255689
time: 17.23 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 8.8992863, 11.2241116, 8.8823147, 11.2298260, -1.9358692, 1.9346869
1: -19.2491074, -15.4888048, -19.2950668, -15.4667673, -3.2236481, 3.2581022
2: -3.8379967, -1.0480481, -3.8283353, -1.0545851, -2.2589898, 2.2483766
3: -13.5404644, -10.0399332, -13.5473003, -10.0233955, -2.7143402, 2.7622881
4: -15.5738430, -12.0072346, -15.5946226, -11.9955750, -2.7358761, 2.7539530
5: -6.1104951, -3.7488642, -6.1168470, -3.7478752, -1.7439768, 1.7530210
6: -3.6106095, -1.4266739, -3.6684361, -1.3857559, -1.9035234, 1.9052432
7: -7.6639819, -3.9012637, -7.7089362, -3.8411403, -3.6189656, 3.6442938
8: -2.8004494, -0.6195374, -2.7906046, -0.6106958, -2.0372992, 2.0204263
9: -9.3933258, -6.0077038, -9.3911753, -6.0158834, -2.5164094, 2.5092349

Time for backsubstitution: 14.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 4568
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 4631
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 52

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 820

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4568

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9357181, upper bound: 0.9216451
time: 4.72 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9368734, upper bound: 0.9216455
time: 5.95 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 8.8992863, 11.2241116, 8.8561115, 11.2455378, -1.9427233, 1.9579663
1: -19.2491074, -15.4888048, -19.3364067, -15.4390163, -3.2556849, 3.2862141
2: -3.8379967, -1.0480481, -3.8488741, -1.0337453, -2.2722788, 2.2683582
3: -13.5404644, -10.0399332, -13.5852375, -9.9870930, -2.7803946, 2.7914104
4: -15.5738430, -12.0072346, -15.6131134, -11.9788380, -2.7491894, 2.7637858
5: -6.1104951, -3.7488642, -6.1280298, -3.7289200, -1.7655535, 1.7677767
6: -3.6106095, -1.4266739, -3.6841607, -1.3765495, -1.9214778, 1.9160564
7: -7.6639819, -3.9012637, -7.7512708, -3.7923930, -3.6591754, 3.6857419
8: -2.8004494, -0.6195374, -2.8119140, -0.5979204, -2.0435133, 2.0353189
9: -9.3933258, -6.0077038, -9.4149847, -5.9927363, -2.5100098, 2.5156205

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 4568
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 4631
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 52

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 820

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4568

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9357174, upper bound: 0.9216451
time: 4.59 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9368737, upper bound: 0.9216470
time: 5.63 seconds

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

Time for backsubstitution: 14.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 4568
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 4631
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 52

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 820

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4568

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9204885, upper bound: 0.9331136
time: 5.27 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9216447, upper bound: 0.9331157
time: 4.52 seconds

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

Time for backsubstitution: 14.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 4568
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 4631
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 52

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 820

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

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
time: 4.70 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 8.8561115, 11.2455378, 8.9226303, 11.2084036, -1.9525685, 1.9280267
1: -19.3364067, -15.4390163, -19.2102623, -15.5165377, -3.2452335, 3.2337642
2: -3.8488741, -1.0337453, -3.8174486, -1.0676342, -2.2568083, 2.2523217
3: -13.5852375, -9.9870930, -13.5026731, -10.0723238, -2.7236366, 2.7431285
4: -15.6131134, -11.9788380, -15.5580549, -12.0238962, -2.7504840, 2.7427964
5: -6.1280298, -3.7289200, -6.1050787, -3.7678983, -1.7460780, 1.7593715
6: -3.6841607, -1.3765495, -3.6007564, -1.4359775, -1.9015379, 1.9129493
7: -7.7512708, -3.7923930, -7.6213474, -3.9494665, -3.6547775, 3.6139655
8: -2.8119140, -0.5979204, -2.7791033, -0.6302538, -2.0311899, 2.0285835
9: -9.4149847, -5.9927363, -9.3696842, -6.0306206, -2.5226712, 2.5035615

Time for backsubstitution: 14.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 4568
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 4631
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 52

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 820

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4568

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9244098, upper bound: 0.9329510
time: 4.71 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9255662, upper bound: 0.9329507
time: 9.81 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 8.8561115, 11.2455378, 8.8992863, 11.2241116, -1.9579663, 1.9427233
1: -19.3364067, -15.4390163, -19.2491074, -15.4888048, -3.2862139, 3.2556849
2: -3.8488741, -1.0337453, -3.8379967, -1.0480481, -2.2683582, 2.2722783
3: -13.5852375, -9.9870930, -13.5404644, -10.0399332, -2.7914104, 2.7803946
4: -15.6131134, -11.9788380, -15.5738430, -12.0072346, -2.7637863, 2.7491899
5: -6.1280298, -3.7289200, -6.1104951, -3.7488642, -1.7677765, 1.7655535
6: -3.6841607, -1.3765495, -3.6106095, -1.4266739, -1.9160566, 1.9214776
7: -7.7512708, -3.7923930, -7.6639819, -3.9012637, -3.6857414, 3.6591759
8: -2.8119140, -0.5979204, -2.8004494, -0.6195374, -2.0353184, 2.0435128
9: -9.4149847, -5.9927363, -9.3933258, -6.0077038, -2.5156202, 2.5100095

Time for backsubstitution: 14.48 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=4, k_high=7, k_mid=5, eps_mid=0.0195312, abs_max=1.955883502960205
rel_dist={0: [-0.9371681487262702, 0.9371704841879342]}

## Binary search (step 2) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5734
type: A, layer: 1, pos: 5758
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 4568
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 4631
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 52

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5734

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8191166, upper bound: 0.8101018
time: 4.89 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8193768, upper bound: 0.8193757
time: 4.77 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 9.82 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 9.82
Output dim: 0, lower bound: -0.8191166, upper bound: 0.8101018
IS_A2, status: Status.UNKNOWN, split count: 1, time: 9.82
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

Time for backsubstitution: 14.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5734
type: B, layer: 1, pos: 5758
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 4568
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 4631
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 52

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5734

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8100995, upper bound: 0.8100996
time: 4.55 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8100995, upper bound: 0.8100997
time: 6.98 seconds

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

Time for backsubstitution: 14.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5734
type: B, layer: 1, pos: 5758
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 4568
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 4631
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 52

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5734

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8100995, upper bound: 0.8191167
time: 4.76 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8100995, upper bound: 0.8193766
time: 6.90 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 26.05 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 26.05
Output dim: 0, lower bound: -0.8100995, upper bound: 0.8100996
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 26.05
Output dim: 0, lower bound: -0.8100995, upper bound: 0.8100997
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 26.05
Output dim: 0, lower bound: -0.8100995, upper bound: 0.8191167
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 26.05
Output dim: 0, lower bound: -0.8100995, upper bound: 0.8193766

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: 8.8562469, 11.2455254, 8.8992767, 11.2241116, -1.8981886, 1.8778391
1: -19.3351517, -15.4390392, -19.2491150, -15.4887924, -3.1895933, 3.1531715
2: -3.8487856, -1.0338333, -3.8380058, -1.0480461, -2.1940956, 2.1980386
3: -13.5839520, -9.9870996, -13.5404816, -10.0399275, -2.6984224, 2.6969168
4: -15.6123123, -11.9788876, -15.5738468, -12.0072260, -2.6554427, 2.6419287
5: -6.1279554, -3.7289562, -6.1104980, -3.7488604, -1.6836872, 1.6815851
6: -3.6841087, -1.3768239, -3.6106141, -1.4266734, -1.8526735, 1.8546848
7: -7.7510715, -3.7930033, -7.6639872, -3.9012482, -3.5705442, 3.5599961
8: -2.8118629, -0.5985670, -2.8004570, -0.6195345, -1.9796867, 1.9873929
9: -9.4144564, -5.9928288, -9.3933372, -6.0077019, -2.4282780, 2.4227111

Time for backsubstitution: 14.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5758
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 4568
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 4631
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 52

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5758

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8069357, upper bound: 0.8191134
time: 5.11 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8100974, upper bound: 0.8191137
time: 5.04 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: 8.8560371, 11.2455463, 8.8560371, 11.2455463, -1.8777337, 1.8777337
1: -19.3369961, -15.4389858, -19.3369961, -15.4389858, -3.1937008, 3.1937003
2: -3.8489280, -1.0337003, -3.8489280, -1.0337003, -2.2055893, 2.2055895
3: -13.5858612, -9.9870853, -13.5858612, -9.9870853, -2.7391887, 2.7391887
4: -15.6134892, -11.9788074, -15.6134892, -11.9788074, -2.6871510, 2.6871514
5: -6.1280670, -3.7288983, -6.1280670, -3.7288983, -1.6936278, 1.6936278
6: -3.6841943, -1.3764181, -3.6841943, -1.3764181, -1.8822737, 1.8822734
7: -7.7513666, -3.7920871, -7.7513666, -3.7920871, -3.6334295, 3.6334298
8: -2.8119488, -0.5976191, -2.8119488, -0.5976191, -1.9914265, 1.9914274
9: -9.4152441, -5.9926920, -9.4152441, -5.9926920, -2.4441619, 2.4441619

Time for backsubstitution: 14.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5758
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 4568
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 4631
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 52

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5758

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8069357, upper bound: 0.8193748
time: 5.22 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8100973, upper bound: 0.8193746
time: 5.99 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 25.64 seconds
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 25.64
Output dim: 0, lower bound: -0.8069357, upper bound: 0.8191134
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 25.64
Output dim: 0, lower bound: -0.8100974, upper bound: 0.8191137
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 25.64
Output dim: 0, lower bound: -0.8069357, upper bound: 0.8193748
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 25.64
Output dim: 0, lower bound: -0.8100973, upper bound: 0.8193746

## BFS IS instance: IS_A2_B1_A1

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

Time for backsubstitution: 14.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5758
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 4568
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 4631
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 52

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5758

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8069357, upper bound: 0.8159554
time: 8.77 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8069357, upper bound: 0.8191134
time: 5.14 seconds

## BFS IS instance: IS_A2_B1_A2

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

Time for backsubstitution: 14.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5758
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 4568
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 4631
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 52

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5758

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8100975, upper bound: 0.8159553
time: 5.26 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8100975, upper bound: 0.8191134
time: 5.13 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: 8.8822489, 11.2298317, 8.8667488, 11.2449074, -1.8496060, 1.8484151
1: -19.2956505, -15.4667501, -19.3321056, -15.4524317, -3.1173019, 3.1383314
2: -3.8283813, -1.0545433, -3.8424397, -1.0366631, -2.1846676, 2.1856091
3: -13.5479069, -10.0233917, -13.5734911, -9.9921799, -2.6380386, 2.6437275
4: -15.5949945, -11.9955511, -15.6114187, -11.9854918, -2.6649070, 2.6707315
5: -6.1168823, -3.7478588, -6.1256814, -3.7323904, -1.6759677, 1.6680472
6: -3.6684625, -1.3856263, -3.6789725, -1.3783405, -1.8740773, 1.8745394
7: -7.7090282, -3.8408487, -7.7467885, -3.8122919, -3.5463858, 3.5661631
8: -2.7906318, -0.6103935, -2.8056364, -0.5990419, -1.9699354, 1.9725919
9: -9.3914280, -6.0158410, -9.4058094, -5.9950023, -2.4339800, 2.4214818

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5758
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 4568
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 4631
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 52

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5758

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8069357, upper bound: 0.8162141
time: 4.77 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8069357, upper bound: 0.8193747
time: 5.08 seconds

## BFS IS instance: IS_A2_B2_A2

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

Time for backsubstitution: 14.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5758
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 4568
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 4631
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 52

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5758

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8100975, upper bound: 0.8162142
time: 5.17 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8100975, upper bound: 0.8193746
time: 4.92 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 24.63 seconds
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 24.63
Output dim: 0, lower bound: -0.8069357, upper bound: 0.8159554
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.63
Output dim: 0, lower bound: -0.8069357, upper bound: 0.8191134
IS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 24.63
Output dim: 0, lower bound: -0.8100975, upper bound: 0.8159553
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.63
Output dim: 0, lower bound: -0.8100975, upper bound: 0.8191134
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 24.63
Output dim: 0, lower bound: -0.8069357, upper bound: 0.8162141
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.63
Output dim: 0, lower bound: -0.8069357, upper bound: 0.8193747
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 24.63
Output dim: 0, lower bound: -0.8100975, upper bound: 0.8162142
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.63
Output dim: 0, lower bound: -0.8100975, upper bound: 0.8193746

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 8.8824577, 11.2298117, 8.8998070, 11.2240839, -1.8574052, 1.8585372
1: -19.2938023, -15.4667988, -19.2488079, -15.4905863, -3.1093259, 3.1021428
2: -3.8282371, -1.0546751, -3.8376904, -1.0489931, -2.1753473, 2.1787183
3: -13.5459995, -10.0234051, -13.5403242, -10.0419312, -2.5991263, 2.6015201
4: -15.5938187, -11.9956284, -15.5735559, -12.0072956, -2.6375809, 2.6243162
5: -6.1167693, -3.7479181, -6.1103330, -3.7489424, -1.6676075, 1.6590459
6: -3.6683784, -1.3860326, -3.6103914, -1.4267349, -1.8397939, 1.8488431
7: -7.7087379, -3.8417661, -7.6623898, -3.9013274, -3.5137196, 3.4670725
8: -2.7905474, -0.6113415, -2.8002443, -0.6197634, -1.9594631, 1.9747133
9: -9.3906355, -6.0159736, -9.3933086, -6.0086851, -2.4184561, 2.4109344

Time for backsubstitution: 14.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 4568
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 4631
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 52

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 820

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4568

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8059425, upper bound: 0.8191102
time: 4.78 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8069325, upper bound: 0.8191102
time: 5.06 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 8.8562574, 11.2455244, 8.8992863, 11.2241116, -1.8798904, 1.8670068
1: -19.3351479, -15.4390545, -19.2491074, -15.4888048, -3.1594076, 3.1333766
2: -3.8487780, -1.0338359, -3.8379967, -1.0480481, -2.1877165, 2.1916344
3: -13.5839357, -9.9871054, -13.5404644, -10.0399332, -2.6823225, 2.6682236
4: -15.6123085, -11.9788942, -15.5738430, -12.0072346, -2.6463223, 2.6325750
5: -6.1279526, -3.7289600, -6.1104951, -3.7488642, -1.6827884, 1.6806867
6: -3.6841030, -1.3768251, -3.6106095, -1.4266739, -1.8505001, 1.8578093
7: -7.7510672, -3.7930179, -7.6639819, -3.9012637, -3.5538912, 3.5238175
8: -2.8118544, -0.5985680, -2.8004494, -0.6195374, -1.9730473, 1.9807529
9: -9.4144440, -5.9928317, -9.3933258, -6.0077038, -2.4084845, 2.4031651

Time for backsubstitution: 14.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 4568
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 4631
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 52

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 820

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4568

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8091192, upper bound: 0.8159525
time: 6.08 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8100942, upper bound: 0.8159523
time: 4.74 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 8.8822489, 11.2298317, 8.8565035, 11.2455177, -1.8503127, 1.8584805
1: -19.2956505, -15.4667501, -19.3367290, -15.4407806, -3.1287613, 3.1426921
2: -3.8283813, -1.0545433, -3.8486099, -1.0346184, -2.1868429, 2.1861589
3: -13.5479069, -10.0233917, -13.5857000, -9.9890060, -2.6205931, 2.6421978
4: -15.5949945, -11.9955511, -15.6132488, -11.9788761, -2.6692066, 2.6697326
5: -6.1168823, -3.7478588, -6.1280146, -3.7289782, -1.6775479, 1.6712110
6: -3.6684625, -1.3856263, -3.6840789, -1.3764801, -1.8714676, 1.8769701
7: -7.7090282, -3.8408487, -7.7497792, -3.7921574, -3.5508909, 3.5406971
8: -2.7906318, -0.6103935, -2.8117390, -0.5978041, -1.9712410, 1.9787521
9: -9.3914280, -6.0158410, -9.4152184, -5.9936743, -2.4343553, 2.4324739

Time for backsubstitution: 14.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 4568
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 4631
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 52

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 820

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8056084, upper bound: 0.8088041
time: 5.33 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8056084, upper bound: 0.8179063
time: 8.14 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 8.8560467, 11.2455463, 8.8560467, 11.2455463, -1.8669000, 1.8669002
1: -19.3369923, -15.4390011, -19.3369923, -15.4390011, -3.1739054, 3.1739058
2: -3.8489184, -1.0337025, -3.8489184, -1.0337025, -2.1992106, 2.1992102
3: -13.5858421, -9.9870930, -13.5858421, -9.9870930, -2.7104955, 2.7104957
4: -15.6134863, -11.9788151, -15.6134863, -11.9788151, -2.6778660, 2.6778660
5: -6.1280642, -3.7289014, -6.1280642, -3.7289014, -1.6927285, 1.6927288
6: -3.6841879, -1.3764200, -3.6841879, -1.3764200, -1.8856411, 1.8856413
7: -7.7513618, -3.7921023, -7.7513618, -3.7921023, -3.5974541, 3.5974541
8: -2.8119426, -0.5976200, -2.8119426, -0.5976200, -1.9847860, 1.9847860
9: -9.4152336, -5.9926934, -9.4152336, -5.9926934, -2.4243689, 2.4243686

Time for backsubstitution: 14.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 4568
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 4631
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 52

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 820

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8087696, upper bound: 0.8056424
time: 4.79 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8087696, upper bound: 0.8147506
time: 7.90 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 27.21 seconds
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 27.21
Output dim: 0, lower bound: -0.8059425, upper bound: 0.8191102
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 27.21
Output dim: 0, lower bound: -0.8069325, upper bound: 0.8191102
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 27.21
Output dim: 0, lower bound: -0.8091192, upper bound: 0.8159525
IS_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 27.21
Output dim: 0, lower bound: -0.8100942, upper bound: 0.8159523
IS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 27.21
Output dim: 0, lower bound: -0.8056084, upper bound: 0.8088041
IS_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 27.21
Output dim: 0, lower bound: -0.8056084, upper bound: 0.8179063
IS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 27.21
Output dim: 0, lower bound: -0.8087696, upper bound: 0.8056424
IS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 27.21
Output dim: 0, lower bound: -0.8087696, upper bound: 0.8147506

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 8.8992376, 11.2220888, 8.9043179, 11.2238722, -1.8398528, 1.8434620
1: -19.2745819, -15.4887838, -19.2434120, -15.4915867, -3.0871840, 3.0762763
2: -3.8102474, -1.0648330, -3.8343987, -1.0506331, -2.1505499, 2.1652570
3: -13.5228462, -10.0448122, -13.5340595, -10.0425987, -2.5639467, 2.5763910
4: -15.5807495, -12.0101614, -15.5728388, -12.0128622, -2.6187253, 2.6064930
5: -6.1094933, -3.7628274, -6.1084728, -3.7501912, -1.6585259, 1.6412969
6: -3.6434548, -1.4035602, -3.6020391, -1.4284272, -1.8139045, 1.8215680
7: -7.6834192, -3.8675454, -7.6596851, -3.9118912, -3.4781046, 3.4393699
8: -2.7801600, -0.6195765, -2.7979412, -0.6203403, -1.9480510, 1.9628758
9: -9.3821526, -6.0257711, -9.3911629, -6.0097628, -2.4069371, 2.3970351

Time for backsubstitution: 14.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 4568
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 4631
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 52

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 820

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 844

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4568

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8059401, upper bound: 0.8181422
time: 5.10 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8059401, upper bound: 0.8191104
time: 4.96 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 8.8824701, 11.2298098, 8.8998117, 11.2240829, -1.8505549, 1.8585305
1: -19.2937889, -15.4668026, -19.2487984, -15.4905872, -3.0980558, 3.1021323
2: -3.8282301, -1.0546765, -3.8376868, -1.0489931, -2.1712065, 2.1785083
3: -13.5459862, -10.0234089, -13.5403194, -10.0419340, -2.5882106, 2.5897236
4: -15.5938187, -11.9956455, -15.5735550, -12.0073032, -2.6367269, 2.6207395
5: -6.1167622, -3.7479212, -6.1103325, -3.7489440, -1.6645310, 1.6590409
6: -3.6683574, -1.3860357, -3.6103842, -1.4267373, -1.8213954, 1.8488305
7: -7.7087312, -3.8417993, -7.6623883, -3.9013438, -3.5105076, 3.4492855
8: -2.7905397, -0.6113443, -2.8002424, -0.6197624, -1.9586887, 1.9747081
9: -9.3906307, -6.0159779, -9.3933077, -6.0086842, -2.4180613, 2.4106250

Time for backsubstitution: 14.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 4568
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 4631
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 52

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 820

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 844

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4568

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8069326, upper bound: 0.8181422
time: 7.37 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8069326, upper bound: 0.8191103
time: 6.57 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 40.21 seconds
IS_A2_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 40.21
Output dim: 0, lower bound: -0.8059401, upper bound: 0.8181422
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 40.21
Output dim: 0, lower bound: -0.8059401, upper bound: 0.8191104
IS_A2_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 40.21
Output dim: 0, lower bound: -0.8069326, upper bound: 0.8181422
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 40.21
Output dim: 0, lower bound: -0.8069326, upper bound: 0.8191103

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 8.8992376, 11.2220888, 8.8998184, 11.2240820, -1.8334925, 1.8485422
1: -19.2745819, -15.4887838, -19.2487965, -15.4905863, -3.0790529, 3.0793142
2: -3.8102474, -1.0648330, -3.8376832, -1.0489969, -2.1518917, 2.1672580
3: -13.5228462, -10.0448122, -13.5403118, -10.0419321, -2.5571947, 2.5719352
4: -15.5807495, -12.0101614, -15.5735531, -12.0073099, -2.6211066, 2.6074018
5: -6.1094933, -3.7628274, -6.1103296, -3.7489448, -1.6595697, 1.6432607
6: -3.6434548, -1.4035602, -3.6103730, -1.4267383, -1.8050244, 1.8299105
7: -7.6834192, -3.8675454, -7.6623840, -3.9013650, -3.4856091, 3.4289083
8: -2.7801600, -0.6195765, -2.8002377, -0.6197634, -1.9487553, 1.9651785
9: -9.3821526, -6.0257711, -9.3933029, -6.0086861, -2.4081264, 2.3982608

Time for backsubstitution: 14.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 4631
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 52

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 820

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 844

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 920

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8045804, upper bound: 0.8191086
time: 4.99 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8059398, upper bound: 0.8191084
time: 4.94 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 8.8824701, 11.2298098, 8.8998184, 11.2240820, -1.8485267, 1.8548942
1: -19.2937889, -15.4668026, -19.2487965, -15.4905863, -3.0965519, 3.0971308
2: -3.8282301, -1.0546765, -3.8376832, -1.0489969, -2.1712046, 2.1792328
3: -13.5459862, -10.0234089, -13.5403118, -10.0419321, -2.5873013, 2.5886965
4: -15.5938187, -11.9956455, -15.5735531, -12.0073099, -2.6339989, 2.6207376
5: -6.1167622, -3.7479212, -6.1103296, -3.7489448, -1.6645298, 1.6559694
6: -3.6683574, -1.3860357, -3.6103730, -1.4267383, -1.8198516, 1.8324139
7: -7.7087312, -3.8417993, -7.6623840, -3.9013650, -3.4995995, 3.4463806
8: -2.7905397, -0.6113443, -2.8002377, -0.6197634, -1.9586868, 1.9739366
9: -9.3906307, -6.0159779, -9.3933029, -6.0086861, -2.4180589, 2.4105382

Time for backsubstitution: 14.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 4631
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 52

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 820

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 844

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 920

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8055705, upper bound: 0.8181397
time: 5.34 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8069300, upper bound: 0.8181398
time: 5.11 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 36.72 seconds
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 36.72
Output dim: 0, lower bound: -0.8045804, upper bound: 0.8191086
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 36.72
Output dim: 0, lower bound: -0.8059398, upper bound: 0.8191084
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 36.72
Output dim: 0, lower bound: -0.8055705, upper bound: 0.8181397
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 36.72
Output dim: 0, lower bound: -0.8069300, upper bound: 0.8181398

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 8.9045811, 11.2211971, 8.9015236, 11.2239285, -1.8263948, 1.8441739
1: -19.2662525, -15.4917850, -19.2461338, -15.4912510, -3.0700669, 3.0736685
2: -3.8101296, -1.0638907, -3.8376455, -1.0493612, -2.1487069, 2.1645818
3: -13.5214767, -10.0475082, -13.5399580, -10.0427952, -2.5533361, 2.5687585
4: -15.5789576, -12.0116749, -15.5729809, -12.0077496, -2.6188288, 2.6042738
5: -6.1077199, -3.7685590, -6.1099486, -3.7507763, -1.6555500, 1.6370556
6: -3.6412234, -1.4038861, -3.6096671, -1.4268191, -1.8028722, 1.8294940
7: -7.6815567, -3.8685122, -7.6617861, -3.9016376, -3.4834175, 3.4264832
8: -2.7770915, -0.6198483, -2.7992687, -0.6197996, -1.9468846, 1.9656091
9: -9.3813896, -6.0263462, -9.3930893, -6.0088758, -2.4056864, 2.3960018

Time for backsubstitution: 14.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 4631
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 52

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 820

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 844

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 920

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8045804, upper bound: 0.8177480
time: 5.50 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8045804, upper bound: 0.8191081
time: 5.64 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 8.8992386, 11.2220898, 8.8998184, 11.2240820, -1.8315339, 1.8485408
1: -19.2745762, -15.4887886, -19.2487965, -15.4905863, -3.0763626, 3.0793104
2: -3.8099236, -1.0648334, -3.8376832, -1.0489969, -2.1494231, 2.1672561
3: -13.5228453, -10.0448112, -13.5403118, -10.0419321, -2.5570598, 2.5719349
4: -15.5807495, -12.0101976, -15.5735531, -12.0073099, -2.6211047, 2.6098657
5: -6.1094923, -3.7628288, -6.1103296, -3.7489448, -1.6595688, 1.6406288
6: -3.6434512, -1.4035614, -3.6103730, -1.4267383, -1.8045270, 1.8299093
7: -7.6834145, -3.8675470, -7.6623840, -3.9013650, -3.4856052, 3.4291759
8: -2.7801552, -0.6195784, -2.8002377, -0.6197634, -1.9529076, 1.9651756
9: -9.3820972, -6.0257769, -9.3933029, -6.0086861, -2.4081254, 2.4005260

Time for backsubstitution: 14.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 4631
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 52

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 820

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 844

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 920

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8059399, upper bound: 0.8177482
time: 5.03 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8059399, upper bound: 0.8191079
time: 4.97 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 36.23 seconds
IS_A2_B1_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 36.23
Output dim: 0, lower bound: -0.8045804, upper bound: 0.8177480
IS_A2_B1_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 36.23
Output dim: 0, lower bound: -0.8045804, upper bound: 0.8191081
IS_A2_B1_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 36.23
Output dim: 0, lower bound: -0.8059399, upper bound: 0.8177482
IS_A2_B1_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 36.23
Output dim: 0, lower bound: -0.8059399, upper bound: 0.8191079

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 8.9045811, 11.2211971, 8.8998194, 11.2240849, -1.8267570, 1.8462162
1: -19.2662525, -15.4917850, -19.2487907, -15.4905949, -3.0699158, 3.0763383
2: -3.8101296, -1.0638907, -3.8373611, -1.0489978, -2.1491814, 2.1654403
3: -13.5214767, -10.0475082, -13.5403061, -10.0419331, -2.5534105, 2.5690422
4: -15.5789576, -12.0116749, -15.5735531, -12.0073490, -2.6190567, 2.6055102
5: -6.1077199, -3.7685590, -6.1103277, -3.7489467, -1.6573634, 1.6375797
6: -3.6412234, -1.4038861, -3.6103711, -1.4267387, -1.8028779, 1.8301508
7: -7.6815567, -3.8685122, -7.6623802, -3.9013665, -3.4836979, 3.4266815
8: -2.7770915, -0.6198483, -2.8002310, -0.6197672, -1.9477773, 1.9653955
9: -9.3813896, -6.0263462, -9.3932438, -6.0086942, -2.4056883, 2.3972976

Time for backsubstitution: 14.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 4631
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 52

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 820

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 844

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4631

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8045786, upper bound: 0.8185609
time: 5.25 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8045786, upper bound: 0.8191067
time: 5.16 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 8.8992386, 11.2220898, 8.8998194, 11.2240849, -1.8315163, 1.8465910
1: -19.2745762, -15.4887886, -19.2487907, -15.4905949, -3.0759325, 3.0765080
2: -3.8099236, -1.0648334, -3.8373611, -1.0489978, -2.1494207, 2.1647868
3: -13.5228453, -10.0448112, -13.5403061, -10.0419331, -2.5570598, 2.5717998
4: -15.5807495, -12.0101976, -15.5735531, -12.0073490, -2.6235700, 2.6098652
5: -6.1094923, -3.7628288, -6.1103277, -3.7489467, -1.6569371, 1.6406274
6: -3.6434512, -1.4035614, -3.6103711, -1.4267387, -1.8045259, 1.8300276
7: -7.6834145, -3.8675470, -7.6623802, -3.9013665, -3.4858742, 3.4291723
8: -2.7801552, -0.6195784, -2.8002310, -0.6197672, -1.9529057, 1.9693270
9: -9.3820972, -6.0257769, -9.3932438, -6.0086942, -2.4103918, 2.4005251

Time for backsubstitution: 14.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 4631
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 52

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 820

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 844

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4631

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8059382, upper bound: 0.8172014
time: 5.06 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8059381, upper bound: 0.8177468
time: 4.85 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 36.04 seconds
IS_A2_B1_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 36.04
Output dim: 0, lower bound: -0.8045786, upper bound: 0.8185609
IS_A2_B1_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 36.04
Output dim: 0, lower bound: -0.8045786, upper bound: 0.8191067
IS_A2_B1_A1_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 36.04
Output dim: 0, lower bound: -0.8059382, upper bound: 0.8172014
IS_A2_B1_A1_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 36.04
Output dim: 0, lower bound: -0.8059381, upper bound: 0.8177468

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 8.9086132, 11.2193613, 8.9006901, 11.2236958, -1.8213582, 1.8420844
1: -19.2376442, -15.5030422, -19.2421837, -15.4926434, -3.0400643, 3.0595551
2: -3.8024335, -1.0727473, -3.8357289, -1.0512440, -2.1384883, 2.1530774
3: -13.4817972, -10.0535927, -13.5309820, -10.0426941, -2.5110188, 2.5356927
4: -15.5745010, -12.0289688, -15.5728369, -12.0110455, -2.6074204, 2.5870223
5: -6.1026745, -3.7723727, -6.1092930, -3.7498083, -1.6502178, 1.6327848
6: -3.6349225, -1.4129243, -3.6092501, -1.4288821, -1.7917318, 1.8212013
7: -7.6627226, -3.8753362, -7.6580162, -3.9025331, -3.4641123, 3.4086518
8: -2.7703476, -0.6245327, -2.7986803, -0.6206303, -1.9407539, 1.9603467
9: -9.3793182, -6.0331459, -9.3929043, -6.0103111, -2.3994217, 2.3889577

Time for backsubstitution: 14.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 4631
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 52

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 820

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 844

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4631

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8040242, upper bound: 0.8185604
time: 5.38 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8040242, upper bound: 0.8185632
time: 9.11 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 8.9045811, 11.2211971, 8.8998194, 11.2240849, -1.8261833, 1.8511004
1: -19.2662506, -15.4917850, -19.2487907, -15.4905949, -3.0564823, 3.0763388
2: -3.8101306, -1.0638924, -3.8373611, -1.0489978, -2.1563873, 2.1654384
3: -13.5214748, -10.0475092, -13.5403061, -10.0419331, -2.5277495, 2.5660200
4: -15.5789566, -12.0116749, -15.5735531, -12.0073490, -2.6188173, 2.5993443
5: -6.1077194, -3.7685595, -6.1103277, -3.7489467, -1.6582375, 1.6375597
6: -3.6412244, -1.4038875, -3.6103711, -1.4267387, -1.8021235, 1.8257341
7: -7.6815538, -3.8685126, -7.6623802, -3.9013665, -3.4766731, 3.4253871
8: -2.7770905, -0.6198483, -2.8002310, -0.6197672, -1.9454079, 1.9653950
9: -9.3813896, -6.0263472, -9.3932438, -6.0086942, -2.4056716, 2.3961482

Time for backsubstitution: 14.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 4631
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 52

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 820

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 844

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4631

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8040242, upper bound: 0.8191063
time: 5.30 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8040242, upper bound: 0.8191069
time: 8.03 seconds

## Summary of splitting at layer (split count: 9)
- Time for IS candidates: 39.76 seconds
IS_A2_B1_A1_B2_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 39.76
Output dim: 0, lower bound: -0.8040242, upper bound: 0.8185604
IS_A2_B1_A1_B2_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 39.76
Output dim: 0, lower bound: -0.8040242, upper bound: 0.8185632
IS_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 39.76
Output dim: 0, lower bound: -0.8040242, upper bound: 0.8191063
IS_A2_B1_A1_B2_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 39.76
Output dim: 0, lower bound: -0.8040242, upper bound: 0.8191069

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 8.9086132, 11.2193613, 8.9039383, 11.2222424, -1.8187339, 1.8381400
1: -19.2376442, -15.5030422, -19.2202015, -15.5018263, -3.0314608, 3.0381103
2: -3.8024335, -1.0727473, -3.8297904, -1.0578630, -2.1316409, 2.1478686
3: -13.4817972, -10.0535927, -13.5005779, -10.0480251, -2.5066109, 2.5221548
4: -15.5745010, -12.0289688, -15.5691977, -12.0246496, -2.5959153, 2.5824103
5: -6.1026745, -3.7723727, -6.1052494, -3.7527373, -1.6475773, 1.6277473
6: -3.6349225, -1.4129243, -3.6040595, -1.4356675, -1.7888210, 1.8161294
7: -7.6627226, -3.8753362, -7.6435566, -3.9081912, -3.4594398, 3.4024081
8: -2.7703476, -0.6245327, -2.7935495, -0.6244183, -1.9378166, 1.9554439
9: -9.3793182, -6.0331459, -9.3912086, -6.0154762, -2.3946872, 2.3863034

Time for backsubstitution: 14.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 52

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 820

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 844

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6137

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8033289, upper bound: 0.8177470
time: 4.86 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8040283, upper bound: 0.8185603
time: 5.44 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 8.9086132, 11.2193613, 8.8999186, 11.2240620, -1.8211091, 1.8426723
1: -19.2376442, -15.5030422, -19.2487965, -15.4906006, -3.0400333, 3.0648437
2: -3.8024335, -1.0727473, -3.8371994, -1.0492488, -2.1418333, 2.1537294
3: -13.4817972, -10.0535927, -13.5401554, -10.0420265, -2.5083756, 2.5378299
4: -15.5745010, -12.0289688, -15.5734682, -12.0069046, -2.6139088, 2.5876675
5: -6.1026745, -3.7723727, -6.1103249, -3.7491045, -1.6507399, 1.6336544
6: -3.6349225, -1.4129243, -3.6103570, -1.4267735, -1.7922544, 1.8222022
7: -7.6627226, -3.8753362, -7.6623950, -3.9013731, -3.4636660, 3.4097285
8: -2.7703476, -0.6245327, -2.8001962, -0.6197929, -1.9413280, 1.9617996
9: -9.3793182, -6.0331459, -9.3932009, -6.0087543, -2.4023232, 2.3894851

Time for backsubstitution: 14.32 seconds
Binary search (step 2): status=Status.UNKNOWN, k_low=4, k_high=4, k_mid=4, eps_mid=0.0156250, abs_max=1.8807220458984375
rel_dist={0: [-0.8193859179682992, 0.8193858024795073]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01171875
execution time: 1665.76 seconds
