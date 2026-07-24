## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 1.1888011651
Search space: {k/256 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.7080693, 3.7080693)
1: (-11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.7309284, 3.7309284)
2: (-10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.9900298, 3.9900298)
3: (-5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.7289009, 2.7289009)
4: (-11.4109173, -8.3298731, -11.4109173, -8.3298731, -3.0810442, 3.0810442)
5: (6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.4367385, 2.4367385)
6: (-8.6112747, -5.0921683, -8.6112747, -5.0921683, -3.5191064, 3.5191064)
7: (-17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.8375950, 3.8375950)
8: (-6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.8985281, 2.8985281)
9: (-4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.4910650, 2.4910650)

## BASE Result
execution time: IAR + LP analysis = 14.68 + 37.44 = 52.11 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -1.8881189, upper bound: 1.8881190


# Binary Search by BASE starts (time budget: 3547.89 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=2.2874808311462402
rel_dist={5: [-1.4581874533617398, 1.4581870675102389]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=2.1325697898864746
rel_dist={5: [-1.1923778842711146, 1.1923772606866638]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=2.0292954444885254
rel_dist={5: [-0.9450640713513714, 0.9450642034848773]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=2.0809326171875
rel_dist={5: [-1.077443187687913, 1.0774448872888032]}

## Binary Search Result
Binary search time: 239.30 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Individual Split (IS_dual_ind) starts
Time budget: 3308.59 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5777
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 4636
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 6136
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 137

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5777

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4946616, upper bound: 1.5314967
time: 14.61 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5369620, upper bound: 1.5369639
time: 14.57 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 29.41 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 29.41
Output dim: 5, lower bound: -1.4946616, upper bound: 1.5314967
IS_A2, status: Status.UNKNOWN, split count: 1, time: 29.41
Output dim: 5, lower bound: -1.5369620, upper bound: 1.5369639

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -9.0259924, -5.4756718, -9.0851383, -5.4095683, -3.4586267, 3.5668802
1: -11.1968470, -7.5448303, -11.2314510, -7.5185065, -3.3152943, 3.3221083
2: -10.3175230, -6.3834968, -10.3379335, -6.3606777, -3.8675318, 3.8345428
3: -5.0175495, -2.3374209, -5.0416355, -2.3243778, -2.6931717, 2.7042146
4: -11.3800459, -8.3609943, -11.4053612, -8.3374720, -2.8899708, 2.8555532
5: 7.0717616, 9.3578358, 6.9946194, 9.3992910, -2.2293406, 2.1745291
6: -8.5192099, -5.1359234, -8.5863113, -5.0954089, -3.1436191, 3.0202205
7: -17.1544342, -13.3804283, -17.1750240, -13.3520775, -3.4517798, 3.5068994
8: -6.0423503, -3.2536507, -6.0789022, -3.2054539, -2.8368964, 2.8195484
9: -4.2012644, -1.7843710, -4.2274771, -1.7510779, -2.4261036, 2.4279099

Time for backsubstitution: 14.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5777
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 4636
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 6136
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 137

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 5777

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4946617, upper bound: 1.4946617
time: 11.09 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4946617, upper bound: 1.5314942
time: 13.89 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -9.0934525, -5.3854008, -9.0934620, -5.3853927, -3.6669102, 3.6660438
1: -11.2401667, -7.5092616, -11.2401772, -7.5092487, -3.3701134, 3.3593974
2: -10.3444233, -6.3544102, -10.3444309, -6.3544011, -3.8770046, 3.8906569
3: -5.0487938, -2.3199053, -5.0488024, -2.3199015, -2.7288923, 2.7288971
4: -11.4109116, -8.3298817, -11.4109173, -8.3298731, -2.9039164, 2.8944755
5: 6.9648223, 9.4015284, 6.9647899, 9.4015284, -2.2879379, 2.3391163
6: -8.6112547, -5.0921721, -8.6112747, -5.0921683, -3.2140975, 3.2394040
7: -17.1788940, -13.3413172, -17.1788979, -13.3413029, -3.5518188, 3.5511532
8: -6.0857372, -3.1872473, -6.0857444, -3.1872163, -2.8918514, 2.8681214
9: -4.2306385, -1.7395942, -4.2306423, -1.7395773, -2.4764977, 2.4623790

Time for backsubstitution: 14.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5777
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 4636
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 6136
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 137

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 5777

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5314948, upper bound: 1.4946613
time: 11.78 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5314946, upper bound: 1.5369638
time: 10.22 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 36.79 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 36.79
Output dim: 5, lower bound: -1.4946617, upper bound: 1.4946617
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 36.79
Output dim: 5, lower bound: -1.4946617, upper bound: 1.5314942
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 36.79
Output dim: 5, lower bound: -1.5314948, upper bound: 1.4946613
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 36.79
Output dim: 5, lower bound: -1.5314946, upper bound: 1.5369638

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -9.0259924, -5.4756718, -9.0259924, -5.4756718, -3.3913403, 3.3913407
1: -11.1968470, -7.5448303, -11.1968470, -7.5448303, -3.2842560, 3.2842560
2: -10.3175230, -6.3834968, -10.3175230, -6.3834968, -3.8395462, 3.8395457
3: -5.0175495, -2.3374209, -5.0175495, -2.3374209, -2.6801286, 2.6801286
4: -11.3800459, -8.3609943, -11.3800459, -8.3609943, -2.8652425, 2.8652425
5: 7.0717616, 9.3578358, 7.0717616, 9.3578358, -2.0977526, 2.0977526
6: -8.5192099, -5.1359234, -8.5192099, -5.1359234, -2.9523993, 2.9523993
7: -17.1544342, -13.3804283, -17.1544342, -13.3804283, -3.4226713, 3.4226718
8: -6.0423503, -3.2536507, -6.0423503, -3.2536507, -2.7886996, 2.7886996
9: -4.2012644, -1.7843710, -4.2012644, -1.7843710, -2.3920593, 2.3920593

Time for backsubstitution: 14.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 4636
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 6136
type: A, layer: 1, pos: 137

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 444

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4938664, upper bound: 1.4832534
time: 10.64 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4945672, upper bound: 1.4945836
time: 17.80 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -9.0259924, -5.4756718, -9.0934525, -5.3854008, -3.4630194, 3.5747118
1: -11.1968470, -7.5448303, -11.2401667, -7.5092616, -3.3249159, 3.3298750
2: -10.3175230, -6.3834968, -10.3444233, -6.3544102, -3.8686228, 3.8450322
3: -5.0175495, -2.3374209, -5.0487938, -2.3199053, -2.6976442, 2.7113729
4: -11.3800459, -8.3609943, -11.4109116, -8.3298817, -2.8980150, 2.8609357
5: 7.0717616, 9.3578358, 6.9648223, 9.4015284, -2.2314906, 2.2035604
6: -8.5192099, -5.1359234, -8.6112547, -5.0921721, -3.1460915, 3.0458632
7: -17.1544342, -13.3804283, -17.1788940, -13.3413172, -3.4638300, 3.5094514
8: -6.0423503, -3.2536507, -6.0857372, -3.1872473, -2.8551030, 2.8254449
9: -4.2012644, -1.7843710, -4.2306385, -1.7395942, -2.4375463, 2.4313037

Time for backsubstitution: 15.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 4636
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 6136
type: A, layer: 1, pos: 137

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 444

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4938664, upper bound: 1.5200721
time: 9.61 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4945671, upper bound: 1.5314042
time: 10.26 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -9.0934525, -5.3854008, -9.0259924, -5.4756718, -3.5747118, 3.4630198
1: -11.2401667, -7.5092616, -11.1968470, -7.5448303, -3.3298759, 3.3249159
2: -10.3444233, -6.3544102, -10.3175230, -6.3834968, -3.8450317, 3.8686233
3: -5.0487938, -2.3199053, -5.0175495, -2.3374209, -2.7113729, 2.6976442
4: -11.4109116, -8.3298817, -11.3800459, -8.3609943, -2.8609357, 2.8980155
5: 6.9648223, 9.4015284, 7.0717616, 9.3578358, -2.2035604, 2.2314904
6: -8.6112547, -5.0921721, -8.5192099, -5.1359234, -3.0458632, 3.1460919
7: -17.1788940, -13.3413172, -17.1544342, -13.3804283, -3.5094509, 3.4638305
8: -6.0857372, -3.1872473, -6.0423503, -3.2536507, -2.8254442, 2.8551030
9: -4.2306385, -1.7395942, -4.2012644, -1.7843710, -2.4313040, 2.4375467

Time for backsubstitution: 14.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 4636
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 6136
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 137

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6182

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5225150, upper bound: 1.4946357
time: 28.69 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5314640, upper bound: 1.4946349
time: 23.63 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -9.0934525, -5.3854008, -9.0934525, -5.3854008, -3.6660357, 3.6660357
1: -11.2401667, -7.5092616, -11.2401667, -7.5092616, -3.3593826, 3.3593826
2: -10.3444233, -6.3544102, -10.3444233, -6.3544102, -3.8906479, 3.8906474
3: -5.0487938, -2.3199053, -5.0487938, -2.3199053, -2.7288885, 2.7288885
4: -11.4109116, -8.3298817, -11.4109116, -8.3298817, -2.9039030, 2.9039035
5: 6.9648223, 9.4015284, 6.9648223, 9.4015284, -2.2879367, 2.2879364
6: -8.6112547, -5.0921721, -8.6112547, -5.0921721, -3.2140927, 3.2140930
7: -17.1788940, -13.3413172, -17.1788940, -13.3413172, -3.5511494, 3.5511494
8: -6.0857372, -3.1872473, -6.0857372, -3.1872473, -2.8681154, 2.8681152
9: -4.2306385, -1.7395942, -4.2306385, -1.7395942, -2.4623752, 2.4623752

Time for backsubstitution: 14.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 4636
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 6136
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 137

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6182

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5225169, upper bound: 1.5001456
time: 5.88 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5314660, upper bound: 1.5001449
time: 9.06 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 29.81 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 29.81
Output dim: 5, lower bound: -1.4938664, upper bound: 1.4832534
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 29.81
Output dim: 5, lower bound: -1.4945672, upper bound: 1.4945836
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 29.81
Output dim: 5, lower bound: -1.4938664, upper bound: 1.5200721
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 29.81
Output dim: 5, lower bound: -1.4945671, upper bound: 1.5314042
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 29.81
Output dim: 5, lower bound: -1.5225150, upper bound: 1.4946357
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 29.81
Output dim: 5, lower bound: -1.5314640, upper bound: 1.4946349
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 29.81
Output dim: 5, lower bound: -1.5225169, upper bound: 1.5001456
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 29.81
Output dim: 5, lower bound: -1.5314660, upper bound: 1.5001449

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -9.0013790, -5.4825554, -9.0203257, -5.4764819, -3.3663368, 3.3787346
1: -11.1801167, -7.5535984, -11.1928864, -7.5460401, -3.2671871, 3.2694864
2: -10.2920351, -6.3986568, -10.3113079, -6.3858347, -3.8127050, 3.8176608
3: -4.9521823, -2.3635240, -5.0002556, -2.3394184, -2.6127639, 2.6367316
4: -11.3687563, -8.3737907, -11.3778887, -8.3637505, -2.8526449, 2.8504353
5: 7.0881395, 9.3252506, 7.0733142, 9.3489466, -2.0713091, 2.0634317
6: -8.5015621, -5.1569586, -8.5147877, -5.1379275, -2.9329295, 2.9210384
7: -17.1051235, -13.4055948, -17.1415749, -13.3825779, -3.3718910, 3.3864713
8: -6.0284562, -3.2596955, -6.0395026, -3.2545214, -2.7739348, 2.7798071
9: -4.1761999, -1.8141538, -4.1991043, -1.7923610, -2.3575487, 2.3604970

Time for backsubstitution: 14.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 4636
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 6136
type: B, layer: 1, pos: 137

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 6182

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4938575, upper bound: 1.4742844
time: 5.70 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4938576, upper bound: 1.4832265
time: 6.21 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -9.0259867, -5.4756718, -9.0259924, -5.4756718, -3.3984966, 3.3890700
1: -11.1968422, -7.5448308, -11.1968470, -7.5448303, -3.2842522, 3.2887430
2: -10.3175201, -6.3834996, -10.3175230, -6.3834968, -3.8298368, 3.8395443
3: -5.0175347, -2.3374219, -5.0175495, -2.3374209, -2.6581850, 2.6801276
4: -11.3800449, -8.3609972, -11.3800459, -8.3609943, -2.8651080, 2.8679132
5: 7.0717640, 9.3578281, 7.0717616, 9.3578358, -2.0977507, 2.0747249
6: -8.5192032, -5.1359253, -8.5192099, -5.1359234, -2.9523916, 2.9693127
7: -17.1544247, -13.3804264, -17.1544342, -13.3804283, -3.4004793, 3.4226708
8: -6.0423465, -3.2536530, -6.0423503, -3.2536507, -2.7886958, 2.7886972
9: -4.2012620, -1.7843776, -4.2012644, -1.7843710, -2.3955421, 2.3920536

Time for backsubstitution: 14.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 4636
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 6136
type: B, layer: 1, pos: 137

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 444

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4832500, upper bound: 1.4938840
time: 7.07 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4832501, upper bound: 1.4938840
time: 11.80 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -9.0013790, -5.4825554, -9.0877762, -5.3862238, -3.4378581, 3.5621991
1: -11.1801167, -7.5535984, -11.2362499, -7.5104918, -3.3077250, 3.3148532
2: -10.2920351, -6.3986568, -10.3381863, -6.3567948, -3.8417206, 3.8231254
3: -4.9521823, -2.3635240, -5.0314755, -2.3219256, -2.6302567, 2.6679516
4: -11.3687563, -8.3737907, -11.4087715, -8.3326511, -2.8854275, 2.8461881
5: 7.0881395, 9.3252506, 6.9663610, 9.3926525, -2.2050118, 2.1692715
6: -8.5015621, -5.1569586, -8.6068115, -5.0942087, -3.1265841, 3.0143781
7: -17.1051235, -13.4055948, -17.1660576, -13.3434973, -3.4129748, 3.4727473
8: -6.0284562, -3.2596955, -6.0829682, -3.1881089, -2.8403473, 2.8167231
9: -4.1761999, -1.8141538, -4.2285013, -1.7475731, -2.4030461, 2.3997636

Time for backsubstitution: 14.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 4636
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 6136
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 137

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 6182

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4938395, upper bound: 1.5111164
time: 5.57 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4938395, upper bound: 1.5200412
time: 5.43 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -9.0259867, -5.4756718, -9.0934525, -5.3854008, -3.4659691, 3.5724406
1: -11.1968422, -7.5448308, -11.2401667, -7.5092616, -3.3249121, 3.3343620
2: -10.3175201, -6.3834996, -10.3444233, -6.3544102, -3.8589153, 3.8450303
3: -5.0175347, -2.3374219, -5.0487938, -2.3199053, -2.6800542, 2.7113719
4: -11.3800449, -8.3609972, -11.4109116, -8.3298817, -2.8978815, 2.8635993
5: 7.0717640, 9.3578281, 6.9648223, 9.4015284, -2.2314878, 2.1799307
6: -8.5192032, -5.1359253, -8.6112547, -5.0921721, -3.1460838, 3.0575094
7: -17.1544247, -13.3804264, -17.1788940, -13.3413172, -3.4416389, 3.5094495
8: -6.0423465, -3.2536530, -6.0857372, -3.1872473, -2.8550992, 2.8254440
9: -4.2012620, -1.7843776, -4.2306385, -1.7395942, -2.4410281, 2.4312975

Time for backsubstitution: 14.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 4636
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 6136
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 137

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 6182

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4945407, upper bound: 1.5224107
time: 7.22 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4945406, upper bound: 1.5313732
time: 10.56 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -9.0801096, -5.3896151, -9.0259924, -5.4756718, -3.5724277, 3.4582598
1: -11.2330971, -7.5143118, -11.1968470, -7.5448303, -3.2897258, 3.3176265
2: -10.3408566, -6.3596568, -10.3175230, -6.3834968, -3.8352089, 3.8589678
3: -5.0467081, -2.3295245, -5.0175495, -2.3374209, -2.7092872, 2.6880250
4: -11.4029036, -8.3309460, -11.3800459, -8.3609943, -2.8516240, 2.8868117
5: 6.9762149, 9.3975840, 7.0717616, 9.3578358, -2.1917658, 2.2333751
6: -8.6063137, -5.0953093, -8.5192099, -5.1359234, -3.0398722, 3.1579759
7: -17.1710320, -13.3429394, -17.1544342, -13.3804283, -3.4997044, 3.4712420
8: -6.0810690, -3.2015038, -6.0423503, -3.2536507, -2.8201637, 2.8408465
9: -4.2248249, -1.7442536, -4.2012644, -1.7843710, -2.4383159, 2.4300230

Time for backsubstitution: 14.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 4636
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 6136
type: B, layer: 1, pos: 137

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 444

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5111165, upper bound: 1.4938388
time: 9.63 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5224109, upper bound: 1.4945400
time: 10.49 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -9.1159449, -5.3056517, -9.0259848, -5.4756742, -3.5960312, 3.5015876
1: -11.2878428, -7.4912629, -11.1968422, -7.5448360, -3.3766980, 3.3405867
2: -10.3582697, -6.2984486, -10.3175201, -6.3834991, -3.8805246, 3.9230518
3: -5.0860376, -2.3069687, -5.0175490, -2.3374279, -2.7486098, 2.7105803
4: -11.4255095, -8.2950697, -11.3800392, -8.3609962, -2.8953719, 2.9400499
5: 6.9530535, 9.4443016, 7.0717735, 9.3578329, -2.2172771, 2.2596574
6: -8.6354847, -5.0771074, -8.5192051, -5.1359262, -3.0751963, 3.1578889
7: -17.2002563, -13.3089342, -17.1544323, -13.3804283, -3.5403433, 3.5059066
8: -6.1710987, -3.1754017, -6.0423455, -3.2536640, -2.9174347, 2.8669438
9: -4.2536182, -1.7114251, -4.2012582, -1.7843752, -2.4644833, 2.4629800

Time for backsubstitution: 14.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 4636
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 6136
type: B, layer: 1, pos: 137

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 444

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5200414, upper bound: 1.4938415
time: 8.86 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5313739, upper bound: 1.4945404
time: 9.20 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -9.0801096, -5.3896151, -9.0934525, -5.3854008, -3.6636906, 3.6613574
1: -11.2330971, -7.5143118, -11.2401667, -7.5092616, -3.3193169, 3.3520856
2: -10.3408566, -6.3596568, -10.3444233, -6.3544102, -3.8808270, 3.8817720
3: -5.0467081, -2.3295245, -5.0487938, -2.3199053, -2.7268028, 2.7192693
4: -11.4029036, -8.3309460, -11.4109116, -8.3298817, -2.8945875, 2.8921952
5: 6.9762149, 9.3975840, 6.9648223, 9.4015284, -2.2765641, 2.2898662
6: -8.6063137, -5.0953093, -8.6112547, -5.0921721, -3.2052388, 3.2259784
7: -17.1710320, -13.3429394, -17.1788940, -13.3413172, -3.5414047, 3.5432787
8: -6.0810690, -3.2015038, -6.0857372, -3.1872473, -2.8628397, 2.8542793
9: -4.2248249, -1.7442536, -4.2306385, -1.7395942, -2.4693041, 2.4554701

Time for backsubstitution: 14.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 4636
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 6136
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 137

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 6182

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5279892, upper bound: 1.4911899
time: 10.56 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5279892, upper bound: 1.5001448
time: 11.77 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -9.1159449, -5.3056517, -9.0934448, -5.3854065, -3.6873522, 3.7067027
1: -11.2878428, -7.4912629, -11.2401638, -7.5092664, -3.4133701, 3.3750548
2: -10.3582697, -6.2984486, -10.3444176, -6.3544140, -3.9261427, 3.9460964
3: -5.0860376, -2.3069687, -5.0487895, -2.3199134, -2.7661242, 2.7418208
4: -11.4255095, -8.2950697, -11.4109068, -8.3298836, -2.9383335, 2.9468904
5: 6.9530535, 9.4443016, 6.9648309, 9.4015245, -2.3053396, 2.3323774
6: -8.6354847, -5.0771074, -8.6112518, -5.0921741, -3.2563992, 3.2258999
7: -17.2002563, -13.3089342, -17.1788883, -13.3413172, -3.5820436, 3.5941672
8: -6.1710987, -3.1754017, -6.0857315, -3.1872597, -2.9635563, 2.8828223
9: -4.2536182, -1.7114251, -4.2306337, -1.7395989, -2.4955616, 2.4885173

Time for backsubstitution: 14.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 4636
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 6136
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 137

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 444

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5255406, upper bound: 1.4993664
time: 24.80 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5368950, upper bound: 1.5001175
time: 6.33 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 46.15 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 46.15
Output dim: 5, lower bound: -1.4938575, upper bound: 1.4742844
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 46.15
Output dim: 5, lower bound: -1.4938576, upper bound: 1.4832265
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 46.15
Output dim: 5, lower bound: -1.4832500, upper bound: 1.4938840
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 46.15
Output dim: 5, lower bound: -1.4832501, upper bound: 1.4938840
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 46.15
Output dim: 5, lower bound: -1.4938395, upper bound: 1.5111164
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 46.15
Output dim: 5, lower bound: -1.4938395, upper bound: 1.5200412
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 46.15
Output dim: 5, lower bound: -1.4945407, upper bound: 1.5224107
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 46.15
Output dim: 5, lower bound: -1.4945406, upper bound: 1.5313732
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 46.15
Output dim: 5, lower bound: -1.5111165, upper bound: 1.4938388
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 46.15
Output dim: 5, lower bound: -1.5224109, upper bound: 1.4945400
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 46.15
Output dim: 5, lower bound: -1.5200414, upper bound: 1.4938415
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 46.15
Output dim: 5, lower bound: -1.5313739, upper bound: 1.4945404
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 46.15
Output dim: 5, lower bound: -1.5279892, upper bound: 1.4911899
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 46.15
Output dim: 5, lower bound: -1.5279892, upper bound: 1.5001448
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 46.15
Output dim: 5, lower bound: -1.5255406, upper bound: 1.4993664
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 46.15
Output dim: 5, lower bound: -1.5368950, upper bound: 1.5001175

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -9.0013790, -5.4825554, -9.0069036, -5.4805784, -3.3618679, 3.3769717
1: -11.1801167, -7.5535984, -11.1861324, -7.5512228, -3.2601852, 3.2294183
2: -10.2920351, -6.3986568, -10.3078003, -6.3911824, -3.8027735, 3.8074646
3: -4.9521823, -2.3635240, -4.9981556, -2.3492014, -2.6029809, 2.6346316
4: -11.3687563, -8.3737907, -11.3699455, -8.3648024, -2.8412485, 2.8408160
5: 7.0881395, 9.3252506, 7.0847683, 9.3452549, -2.0612907, 2.0515947
6: -8.5015621, -5.1569586, -8.5098085, -5.1407900, -2.9304295, 2.9150543
7: -17.1051235, -13.4055948, -17.1336422, -13.3841724, -3.3799486, 3.3774943
8: -6.0284562, -3.2596955, -6.0350595, -3.2688451, -2.7596111, 2.7753639
9: -4.1761999, -1.8141538, -4.1935339, -1.7971276, -2.3498416, 2.3637171

Time for backsubstitution: 14.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 4636
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 6136
type: A, layer: 1, pos: 137

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6182

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4849148, upper bound: 1.4742844
time: 5.44 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4849148, upper bound: 1.4742844
time: 8.52 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -9.0013714, -5.4825559, -9.0412626, -5.3954687, -3.4289865, 3.3995748
1: -11.1801128, -7.5536046, -11.2404957, -7.5279179, -3.2831488, 3.3200693
2: -10.2920322, -6.3986597, -10.3239536, -6.3286295, -3.8665056, 3.8532634
3: -4.9521804, -2.3635314, -5.0368252, -2.3269439, -2.6252365, 2.6732938
4: -11.3687496, -8.3737936, -11.3910799, -8.3286743, -2.8953896, 2.8905878
5: 7.0881519, 9.3252468, 7.0618052, 9.3896723, -2.1149354, 2.0766385
6: -8.5015564, -5.1569624, -8.5389938, -5.1254072, -2.9474354, 2.9519939
7: -17.1051178, -13.4055996, -17.1618557, -13.3505507, -3.4124565, 3.4347429
8: -6.0284538, -3.2597089, -6.1205168, -3.2431421, -2.7853117, 2.8608079
9: -4.1761951, -1.8141583, -4.2212963, -1.7642260, -2.3832693, 2.3944423

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 4636
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 6136
type: A, layer: 1, pos: 137

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 863

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6182

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4849150, upper bound: 1.4832271
time: 5.36 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4849150, upper bound: 1.4832270
time: 8.17 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -9.0251760, -5.4758577, -9.0013790, -5.4825554, -3.3796959, 3.3647408
1: -11.1965961, -7.5453949, -11.1801167, -7.5535984, -3.2733593, 3.2658176
2: -10.3155899, -6.3836946, -10.2920351, -6.3986568, -3.8230286, 3.8147697
3: -5.0172224, -2.3376145, -4.9521823, -2.3635240, -2.6536984, 2.6145678
4: -11.3794250, -8.3622084, -11.3687563, -8.3737907, -2.8523846, 2.8524704
5: 7.0717797, 9.3574066, 7.0881395, 9.3252506, -2.0645032, 2.0802736
6: -8.5188103, -5.1382871, -8.5015621, -5.1569586, -2.9260397, 2.9279759
7: -17.1542740, -13.3807936, -17.1051235, -13.4055948, -3.3991427, 3.3727741
8: -6.0422854, -3.2538652, -6.0284562, -3.2596955, -2.7825899, 2.7745910
9: -4.2002401, -1.7845263, -4.1761999, -1.8141538, -2.3604565, 2.3660011

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 4636
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 6136
type: A, layer: 1, pos: 137

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6182

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4742805, upper bound: 1.4938590
time: 16.68 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4832227, upper bound: 1.4938567
time: 11.63 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -9.0259867, -5.4756718, -9.0259867, -5.4756718, -3.3984947, 3.3984947
1: -11.1968422, -7.5448308, -11.1968422, -7.5448308, -3.2887383, 3.2887383
2: -10.3175201, -6.3834996, -10.3175201, -6.3834996, -3.8298330, 3.8298340
3: -5.0175347, -2.3374219, -5.0175347, -2.3374219, -2.6581826, 2.6581829
4: -11.3800449, -8.3609972, -11.3800449, -8.3609972, -2.8679094, 2.8679094
5: 7.0717640, 9.3578281, 7.0717640, 9.3578281, -2.0747225, 2.0747225
6: -8.5192032, -5.1359253, -8.5192032, -5.1359253, -2.9693060, 2.9693065
7: -17.1544247, -13.3804264, -17.1544247, -13.3804264, -3.4004774, 3.4004779
8: -6.0423465, -3.2536530, -6.0423465, -3.2536530, -2.7886934, 2.7886934
9: -4.2012620, -1.7843776, -4.2012620, -1.7843776, -2.3955359, 2.3955359

Time for backsubstitution: 14.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 4636
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 6136
type: A, layer: 1, pos: 137

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6182

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4742805, upper bound: 1.4938569
time: 19.10 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4832227, upper bound: 1.4945601
time: 8.67 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -9.0013790, -5.4825554, -9.0744276, -5.3904390, -3.4331002, 3.5599031
1: -11.1801167, -7.5535984, -11.2291794, -7.5155439, -3.3004360, 3.2746868
2: -10.2920351, -6.3986568, -10.3346462, -6.3620400, -3.8320665, 3.8132963
3: -4.9521823, -2.3635240, -5.0293889, -2.3315468, -2.6206355, 2.6658649
4: -11.3687563, -8.3737907, -11.4007664, -8.3337145, -2.8741760, 2.8368759
5: 7.0881395, 9.3252506, 6.9777584, 9.3887091, -2.2068901, 2.1574748
6: -8.5015621, -5.1569586, -8.6018581, -5.0973406, -3.1384678, 3.0083673
7: -17.1051235, -13.4055948, -17.1581879, -13.3451233, -3.4204321, 3.4629908
8: -6.0284562, -3.2596955, -6.0783052, -3.2023692, -2.8260870, 2.8114676
9: -4.1761999, -1.8141538, -4.2226825, -1.7522386, -2.3955159, 2.4067540

Time for backsubstitution: 14.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 4636
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 6136
type: A, layer: 1, pos: 137

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6182

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4848967, upper bound: 1.5111092
time: 5.36 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4848969, upper bound: 1.5111096
time: 7.37 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -9.0013714, -5.4825559, -9.1102486, -5.3064685, -3.4764028, 3.5835533
1: -11.1801128, -7.5536046, -11.2839041, -7.4924965, -3.3233910, 3.3615398
2: -10.2920322, -6.3986597, -10.3520336, -6.3007994, -3.8962097, 3.8586211
3: -4.9521804, -2.3635314, -5.0687275, -2.3089685, -2.6432118, 2.7051961
4: -11.3687496, -8.3737936, -11.4233618, -8.2978249, -2.9257674, 2.8806393
5: 7.0881519, 9.3252468, 6.9545794, 9.4354210, -2.2243745, 2.1829958
6: -8.5015564, -5.1569624, -8.6310539, -5.0791607, -3.1383176, 3.0439606
7: -17.1051178, -13.4055996, -17.1874313, -13.3111048, -3.4550886, 3.5036554
8: -6.0284538, -3.2597089, -6.1683064, -3.1762600, -2.8521938, 2.9085975
9: -4.1761951, -1.8141583, -4.2514954, -1.7194059, -2.4284763, 2.4329543

Time for backsubstitution: 14.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 4636
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 6136
type: A, layer: 1, pos: 137

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 863

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4670604, upper bound: 1.5142912
time: 12.51 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=2.339118003845215
rel_dist={5: [-1.5369944972060878, 1.5369941521841248]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5777
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 4636
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 6136
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 137

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 5777

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2597702, upper bound: 1.2807557
time: 5.44 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2867659, upper bound: 1.2867678
time: 9.08 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 14.75 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 14.75
Output dim: 5, lower bound: -1.2597702, upper bound: 1.2807557
IS_A2, status: Status.UNKNOWN, split count: 1, time: 14.75
Output dim: 5, lower bound: -1.2867659, upper bound: 1.2867678

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -9.0259924, -5.4756718, -9.0796051, -5.4253631, -3.2064128, 3.3317747
1: -11.1968470, -7.5448303, -11.2256804, -7.5245371, -3.0405540, 3.0473013
2: -10.3175230, -6.3834968, -10.3337202, -6.3647361, -3.6589861, 3.6267052
3: -5.0175495, -2.3374209, -5.0369749, -2.3270478, -2.4732161, 2.4894910
4: -11.3800459, -8.3609943, -11.4016581, -8.3424330, -2.6405945, 2.6160450
5: 7.0717616, 9.3578358, 7.0140963, 9.3977842, -2.0729833, 1.9927022
6: -8.5192099, -5.1359234, -8.5700216, -5.0974593, -2.8603177, 2.7177935
7: -17.1544342, -13.3804283, -17.1724262, -13.3590975, -3.1315145, 3.1990480
8: -6.0423503, -3.2536507, -6.0746112, -3.2173529, -2.6722426, 2.6379662
9: -4.2012644, -1.7843710, -4.2253752, -1.7585533, -2.3112373, 2.3200927

Time for backsubstitution: 14.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5777
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 4636
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 6136
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 137

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 5777

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2597703, upper bound: 1.2597725
time: 9.46 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2597703, upper bound: 1.2807560
time: 8.10 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -9.0934525, -5.3854008, -9.0934601, -5.3853960, -3.4362292, 3.4359980
1: -11.2401667, -7.5092616, -11.2401733, -7.5092525, -3.1016350, 3.0872531
2: -10.3444233, -6.3544102, -10.3444300, -6.3544035, -3.6728125, 3.6858082
3: -5.0487938, -2.3199053, -5.0488000, -2.3199024, -2.5250592, 2.5249879
4: -11.4109116, -8.3298817, -11.4109135, -8.3298750, -2.6684704, 2.6594801
5: 6.9648223, 9.4015284, 6.9647994, 9.4015284, -2.1240072, 2.1841950
6: -8.6112547, -5.0921721, -8.6112671, -5.0921702, -2.9279842, 2.9577346
7: -17.1788940, -13.3413172, -17.1788998, -13.3413057, -3.2456646, 3.2448854
8: -6.0857372, -3.1872473, -6.0857406, -3.1872244, -2.7141905, 2.6862781
9: -4.2306385, -1.7395942, -4.2306409, -1.7395815, -2.3709340, 2.3543277

Time for backsubstitution: 14.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5777
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 4636
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 6136
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 137

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 5777

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2807558, upper bound: 1.2597703
time: 14.36 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2807557, upper bound: 1.2597704
time: 6.79 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 36.08 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 36.08
Output dim: 5, lower bound: -1.2597703, upper bound: 1.2597725
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 36.08
Output dim: 5, lower bound: -1.2597703, upper bound: 1.2807560
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 36.08
Output dim: 5, lower bound: -1.2807558, upper bound: 1.2597703
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 36.08
Output dim: 5, lower bound: -1.2807557, upper bound: 1.2597704

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -9.0259924, -5.4756718, -9.0259924, -5.4756718, -3.1574860, 3.1574860
1: -11.1968470, -7.5448303, -11.1968470, -7.5448303, -3.0155706, 3.0155706
2: -10.3175230, -6.3834968, -10.3175230, -6.3834968, -3.6345930, 3.6345925
3: -5.0175495, -2.3374209, -5.0175495, -2.3374209, -2.4608741, 2.4608741
4: -11.3800459, -8.3609943, -11.3800459, -8.3609943, -2.6211181, 2.6211176
5: 7.0717616, 9.3578358, 7.0717616, 9.3578358, -1.9352875, 1.9352870
6: -8.5192099, -5.1359234, -8.5192099, -5.1359234, -2.6667137, 2.6667140
7: -17.1544342, -13.3804283, -17.1544342, -13.3804283, -3.1103678, 3.1103678
8: -6.0423503, -3.2536507, -6.0423503, -3.2536507, -2.6363707, 2.6363707
9: -4.2012644, -1.7843710, -4.2012644, -1.7843710, -2.2846661, 2.2846646

Time for backsubstitution: 14.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 4636
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 6136
type: A, layer: 1, pos: 137

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 444

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2584658, upper bound: 1.2537486
time: 10.01 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2597203, upper bound: 1.2609324
time: 11.70 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -9.0259924, -5.4756718, -9.0934525, -5.3854008, -3.2131157, 3.3439946
1: -11.1968470, -7.5448303, -11.2401667, -7.5092616, -3.0553536, 3.0611897
2: -10.3175230, -6.3834968, -10.3444233, -6.3544102, -3.6636696, 3.6418395
3: -5.0175495, -2.3374209, -5.0487938, -2.3199053, -2.4827447, 2.5034428
4: -11.3800459, -8.3609943, -11.4109116, -8.3298817, -2.6538906, 2.6266332
5: 7.0717616, 9.3578358, 6.9648223, 9.4015284, -2.0765796, 2.0216103
6: -8.5192099, -5.1359234, -8.6112547, -5.0921721, -2.8644295, 2.7460048
7: -17.1544342, -13.3804283, -17.1788940, -13.3413172, -3.1515265, 3.2033014
8: -6.0423503, -3.2536507, -6.0857372, -3.1872473, -2.7021189, 2.6477914
9: -4.2012644, -1.7843710, -4.2306385, -1.7395942, -2.3301530, 2.3257451

Time for backsubstitution: 14.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 4636
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 6136
type: A, layer: 1, pos: 137

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 444

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2584657, upper bound: 1.2735327
time: 8.02 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2597203, upper bound: 1.2807051
time: 17.65 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -9.0934525, -5.3854008, -9.0259924, -5.4756718, -3.3439946, 3.2131157
1: -11.2401667, -7.5092616, -11.1968470, -7.5448303, -3.0611906, 3.0553534
2: -10.3444233, -6.3544102, -10.3175230, -6.3834968, -3.6418400, 3.6636701
3: -5.0487938, -2.3199053, -5.0175495, -2.3374209, -2.5034432, 2.4827442
4: -11.4109116, -8.3298817, -11.3800459, -8.3609943, -2.6266332, 2.6538906
5: 6.9648223, 9.4015284, 7.0717616, 9.3578358, -2.0216103, 2.0765793
6: -8.6112547, -5.0921721, -8.5192099, -5.1359234, -2.7460046, 2.8644297
7: -17.1788940, -13.3413172, -17.1544342, -13.3804283, -3.2033014, 3.1515269
8: -6.0857372, -3.1872473, -6.0423503, -3.2536507, -2.6477919, 2.7021194
9: -4.2306385, -1.7395942, -4.2012644, -1.7843710, -2.3257456, 2.3301520

Time for backsubstitution: 14.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 4636
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 6136
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 137

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6182

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2745694, upper bound: 1.2597617
time: 15.99 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2807434, upper bound: 1.2597619
time: 8.53 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -9.0934525, -5.3854008, -9.0934525, -5.3854008, -3.4355240, 3.4355242
1: -11.2401667, -7.5092616, -11.2401667, -7.5092616, -3.0872412, 3.0872412
2: -10.3444233, -6.3544102, -10.3444233, -6.3544102, -3.6858025, 3.6858020
3: -5.0487938, -2.3199053, -5.0487938, -2.3199053, -2.5250559, 2.5250554
4: -11.4109116, -8.3298817, -11.4109116, -8.3298817, -2.6684618, 2.6684628
5: 6.9648223, 9.4015284, 6.9648223, 9.4015284, -2.1240058, 2.1240063
6: -8.6112547, -5.0921721, -8.6112547, -5.0921721, -2.9279809, 2.9279809
7: -17.1788940, -13.3413172, -17.1788940, -13.3413172, -3.2448826, 3.2448835
8: -6.0857372, -3.1872473, -6.0857372, -3.1872473, -2.6862726, 2.6862729
9: -4.2306385, -1.7395942, -4.2306385, -1.7395942, -2.3543258, 2.3543253

Time for backsubstitution: 14.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 4636
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 6136
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 137

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6182

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2745714, upper bound: 1.2657639
time: 7.08 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2807454, upper bound: 1.2657632
time: 29.79 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 51.85 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 51.85
Output dim: 5, lower bound: -1.2584658, upper bound: 1.2537486
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 51.85
Output dim: 5, lower bound: -1.2597203, upper bound: 1.2609324
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 51.85
Output dim: 5, lower bound: -1.2584657, upper bound: 1.2735327
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 51.85
Output dim: 5, lower bound: -1.2597203, upper bound: 1.2807051
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 51.85
Output dim: 5, lower bound: -1.2745694, upper bound: 1.2597617
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 51.85
Output dim: 5, lower bound: -1.2807434, upper bound: 1.2597619
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 51.85
Output dim: 5, lower bound: -1.2745714, upper bound: 1.2657639
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 51.85
Output dim: 5, lower bound: -1.2807454, upper bound: 1.2657632

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -9.0013790, -5.4825554, -9.0166016, -5.4770236, -3.1319160, 3.1424432
1: -11.1801167, -7.5535984, -11.1903028, -7.5468521, -2.9978724, 2.9987645
2: -10.2920351, -6.3986568, -10.3072882, -6.3873806, -3.6062431, 3.6076059
3: -4.9521823, -2.3635240, -4.9889345, -2.3407564, -2.3928409, 2.4075415
4: -11.3687563, -8.3737907, -11.3764572, -8.3656254, -2.6077337, 2.6047530
5: 7.0881395, 9.3252506, 7.0743361, 9.3431187, -1.9024215, 1.8999243
6: -8.5015621, -5.1569586, -8.5119028, -5.1393356, -2.6458292, 2.6337314
7: -17.1051235, -13.4055948, -17.1331501, -13.3840094, -3.0585623, 3.0657110
8: -6.0284562, -3.2596955, -6.0376158, -3.2550955, -2.6190710, 2.6253686
9: -4.1761999, -1.8141538, -4.1976295, -1.7975876, -2.2445126, 2.2517405

Time for backsubstitution: 14.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 4636
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 6136
type: B, layer: 1, pos: 137

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6182

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2596682, upper bound: 1.2475593
time: 9.97 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2596683, upper bound: 1.2537400
time: 6.35 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -9.0259867, -5.4756718, -9.0259886, -5.4756708, -3.1623850, 3.1538210
1: -11.1968422, -7.5448308, -11.1968460, -7.5448308, -3.0152311, 3.0195131
2: -10.3175201, -6.3834996, -10.3175201, -6.3834963, -3.6227617, 3.6345882
3: -5.0175347, -2.3374219, -5.0175452, -2.3374209, -2.4228802, 2.4608691
4: -11.3800449, -8.3609972, -11.3800459, -8.3609962, -2.6209822, 2.6223111
5: 7.0717640, 9.3578281, 7.0717621, 9.3578329, -1.9352849, 1.9077904
6: -8.5192032, -5.1359253, -8.5192070, -5.1359243, -2.6654387, 2.6815739
7: -17.1544247, -13.3804264, -17.1544342, -13.3804264, -3.0842676, 3.1103640
8: -6.0423465, -3.2536530, -6.0423489, -3.2536516, -2.6357951, 2.6359031
9: -4.2012620, -1.7843776, -4.2012615, -1.7843720, -2.2877231, 2.2843976

Time for backsubstitution: 14.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 4636
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 6136
type: B, layer: 1, pos: 137

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 6182

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2609244, upper bound: 1.2547420
time: 5.36 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2609245, upper bound: 1.2609232
time: 10.34 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -9.0013790, -5.4825554, -9.0840464, -5.3867688, -3.1873794, 3.3287277
1: -11.1801167, -7.5535984, -11.2336979, -7.5113606, -3.0375743, 3.0442185
2: -10.2920351, -6.3986568, -10.3341217, -6.3583674, -3.6352186, 3.6148291
3: -4.9521823, -2.3635240, -5.0201378, -2.3232784, -2.4146495, 2.4501631
4: -11.3687563, -8.3737907, -11.4073496, -8.3345346, -2.6404996, 2.6103683
5: 7.0881395, 9.3252506, 6.9673729, 9.3868380, -2.0436177, 1.9862765
6: -8.5015621, -5.1569586, -8.6039076, -5.0956383, -2.8435116, 2.7132134
7: -17.1051235, -13.4055948, -17.1576500, -13.3449478, -3.0995951, 3.1581464
8: -6.0284562, -3.2596955, -6.0811372, -3.1886787, -2.6848354, 2.6373162
9: -4.1761999, -1.8141538, -4.2270417, -1.7527925, -2.2900162, 2.2928371

Time for backsubstitution: 14.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 4636
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 6136
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 137

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 6182

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2584572, upper bound: 1.2673479
time: 11.92 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2584572, upper bound: 1.2735203
time: 9.49 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -9.0259867, -5.4756718, -9.0934534, -5.3854012, -3.2138004, 3.3380744
1: -11.1968422, -7.5448308, -11.2401676, -7.5092597, -3.0542173, 3.0651317
2: -10.3175201, -6.3834996, -10.3444195, -6.3544092, -3.6518402, 3.6418357
3: -5.0175347, -2.3374219, -5.0487885, -2.3199081, -2.4447508, 2.5034380
4: -11.3800449, -8.3609972, -11.4109116, -8.3298836, -2.6537552, 2.6278210
5: 7.0717640, 9.3578281, 6.9648218, 9.4015245, -2.0765758, 1.9932517
6: -8.5192032, -5.1359253, -8.6112537, -5.0921736, -2.8631549, 2.7509162
7: -17.1544247, -13.3804264, -17.1788902, -13.3413153, -3.1254253, 3.2032967
8: -6.0423465, -3.2536530, -6.0857363, -3.1872478, -2.7015448, 2.6473250
9: -4.2012620, -1.7843776, -4.2306390, -1.7395961, -2.3332090, 2.3254786

Time for backsubstitution: 14.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 4636
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 6136
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 137

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 6182

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2597117, upper bound: 1.2745186
time: 6.14 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2597117, upper bound: 1.2806926
time: 13.17 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -9.0801096, -5.3896151, -9.0225172, -5.4767809, -3.3406086, 3.2041824
1: -11.2330971, -7.5143118, -11.1949425, -7.5461593, -3.0190363, 3.0458586
2: -10.3408566, -6.3596568, -10.3165836, -6.3849502, -3.6288853, 3.6516137
3: -5.0467081, -2.3295245, -5.0169716, -2.3399370, -2.4969974, 2.4703896
4: -11.4029036, -8.3309460, -11.3779831, -8.3613119, -2.6169620, 2.6389804
5: 6.9762149, 9.3975840, 7.0747018, 9.3568535, -2.0090184, 2.0731261
6: -8.6063137, -5.0953093, -8.5178995, -5.1368337, -2.7384992, 2.8708386
7: -17.1710320, -13.3429394, -17.1522789, -13.3809681, -3.1929655, 3.1575236
8: -6.0810690, -3.2015038, -6.0411282, -3.2573295, -2.6370225, 2.6864867
9: -4.2248249, -1.7442536, -4.1997652, -1.7856108, -2.3317404, 2.3206100

Time for backsubstitution: 14.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 4636
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 6136
type: B, layer: 1, pos: 137

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 444

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2673462, upper bound: 1.2584571
time: 10.68 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2745186, upper bound: 1.2597118
time: 9.00 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -9.1158428, -5.3056645, -9.0259790, -5.4756765, -3.3643603, 3.2519755
1: -11.2876425, -7.4912691, -11.1968355, -7.5448389, -3.1028819, 3.0714765
2: -10.3582659, -6.2984757, -10.3175182, -6.3835030, -3.6743832, 3.7180529
3: -5.0860233, -2.3069980, -5.0175467, -2.3374338, -2.5418053, 2.5088542
4: -11.4253759, -8.2950745, -11.3800364, -8.3609972, -2.6588736, 2.6896980
5: 6.9530730, 9.4442015, 7.0717802, 9.3578310, -2.0323415, 2.0994666
6: -8.6354637, -5.0771637, -8.5191994, -5.1359272, -2.7692957, 2.8761590
7: -17.2001114, -13.3089409, -17.1544304, -13.3804283, -3.2334309, 3.1935921
8: -6.1710625, -3.1754260, -6.0423441, -3.2536721, -2.7377234, 2.7108269
9: -4.2534437, -1.7114313, -4.2012553, -1.7843777, -2.3578997, 2.3555727

Time for backsubstitution: 14.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 4636
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 6136
type: B, layer: 1, pos: 137

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 444

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2735207, upper bound: 1.2584576
time: 7.58 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2806926, upper bound: 1.2597117
time: 11.10 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -9.0801096, -5.3896151, -9.0899992, -5.3865418, -3.4319029, 3.4267688
1: -11.2330971, -7.5143118, -11.2381840, -7.5105562, -3.0452051, 3.0777841
2: -10.3408566, -6.3596568, -10.3435030, -6.3558292, -3.6728945, 3.6745310
3: -5.0467081, -2.3295245, -5.0482168, -2.3223817, -2.5185990, 2.5124676
4: -11.4029036, -8.3309460, -11.4088335, -8.3302031, -2.6587820, 2.6529760
5: 6.9762149, 9.3975840, 6.9677525, 9.4004784, -2.1115155, 2.1205857
6: -8.6063137, -5.0953093, -8.6099625, -5.0931506, -2.9168401, 2.9343796
7: -17.1710320, -13.3429394, -17.1768303, -13.3418684, -3.2345209, 3.2361941
8: -6.0810690, -3.2015038, -6.0844612, -3.1909113, -2.6755171, 2.6711109
9: -4.2248249, -1.7442536, -4.2290778, -1.7408144, -2.3602738, 2.3454537

Time for backsubstitution: 14.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 4636
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 6136
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 137

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 444

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2463478, upper bound: 1.2644154
time: 5.80 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2805248, upper bound: 1.2657137
time: 12.11 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -9.1158466, -5.3056645, -9.0934401, -5.3854074, -3.4556971, 3.4726663
1: -11.2876444, -7.4912672, -11.2401590, -7.5092697, -3.1360092, 3.1028981
2: -10.3582668, -6.2984753, -10.3444147, -6.3544188, -3.7183456, 3.7412033
3: -5.0860243, -2.3069968, -5.0487905, -2.3199182, -2.5634174, 2.5518849
4: -11.4253778, -8.2950735, -11.4109011, -8.3298836, -2.7007008, 2.7114401
5: 6.9530721, 9.4442024, 6.9648385, 9.4015217, -2.1368544, 2.1653516
6: -8.6354637, -5.0771623, -8.6112442, -5.0921764, -2.9672632, 2.9397202
7: -17.2001114, -13.3089437, -17.1788845, -13.3413181, -3.2750149, 3.2878890
8: -6.1710625, -3.1754265, -6.0857310, -3.1872687, -2.7816372, 2.6969168
9: -4.2534447, -1.7114320, -4.2306304, -1.7396005, -2.3864870, 2.3804564

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 4636
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 6136
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 137

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 444

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2794755, upper bound: 1.2644149
time: 11.12 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2866938, upper bound: 1.2657136
time: 11.72 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 37.59 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 37.59
Output dim: 5, lower bound: -1.2596682, upper bound: 1.2475593
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 37.59
Output dim: 5, lower bound: -1.2596683, upper bound: 1.2537400
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 37.59
Output dim: 5, lower bound: -1.2609244, upper bound: 1.2547420
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 37.59
Output dim: 5, lower bound: -1.2609245, upper bound: 1.2609232
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 37.59
Output dim: 5, lower bound: -1.2584572, upper bound: 1.2673479
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 37.59
Output dim: 5, lower bound: -1.2584572, upper bound: 1.2735203
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 37.59
Output dim: 5, lower bound: -1.2597117, upper bound: 1.2745186
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 37.59
Output dim: 5, lower bound: -1.2597117, upper bound: 1.2806926
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 37.59
Output dim: 5, lower bound: -1.2673462, upper bound: 1.2584571
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 37.59
Output dim: 5, lower bound: -1.2745186, upper bound: 1.2597118
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 37.59
Output dim: 5, lower bound: -1.2735207, upper bound: 1.2584576
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 37.59
Output dim: 5, lower bound: -1.2806926, upper bound: 1.2597117
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 37.59
Output dim: 5, lower bound: -1.2463478, upper bound: 1.2644154
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 37.59
Output dim: 5, lower bound: -1.2805248, upper bound: 1.2657137
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 37.59
Output dim: 5, lower bound: -1.2794755, upper bound: 1.2644149
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 37.59
Output dim: 5, lower bound: -1.2866938, upper bound: 1.2657136

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -8.9979029, -5.4836698, -9.0031796, -5.4811220, -3.1237259, 3.1392589
1: -11.1782112, -7.5549293, -11.1835489, -7.5520391, -2.9887772, 2.9567018
2: -10.2911339, -6.4001079, -10.3037806, -6.3927259, -3.5939112, 3.5941148
3: -4.9516001, -2.3660431, -4.9868331, -2.3505416, -2.3805218, 2.4012425
4: -11.3666964, -8.3741102, -11.3685160, -8.3666801, -2.5926332, 2.5945034
5: 7.0910830, 9.3242636, 7.0857887, 9.3394251, -1.8883359, 1.8872862
6: -8.5002728, -5.1578631, -8.5069132, -5.1421967, -2.6392841, 2.6264079
7: -17.1029530, -13.4061375, -17.1252174, -13.3856030, -3.0652218, 3.0560265
8: -6.0272388, -3.2633767, -6.0331736, -3.2694197, -2.6033831, 2.6113107
9: -4.1746998, -1.8153970, -4.1920586, -1.8023579, -2.2347803, 2.2540135

Time for backsubstitution: 14.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 4636
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 6136
type: A, layer: 1, pos: 137

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 863

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6182

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2534900, upper bound: 1.2475597
time: 8.13 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2534900, upper bound: 1.2475600
time: 44.85 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -9.0013676, -5.4825592, -9.0374374, -5.3960123, -3.1908092, 3.1602464
1: -11.1801100, -7.5536098, -11.2376957, -7.5287237, -3.0138168, 3.0442753
2: -10.2920294, -6.3986626, -10.3199329, -6.3302307, -3.6599693, 3.6399698
3: -4.9521794, -2.3635368, -5.0254927, -2.3282957, -2.4173641, 2.4450624
4: -11.3687449, -8.3737926, -11.3895283, -8.3305483, -2.6504583, 2.6417961
5: 7.0881586, 9.3252439, 7.0628376, 9.3837395, -1.9459407, 1.9101453
6: -8.5015545, -5.1569633, -8.5360928, -5.1268702, -2.6602387, 2.6632557
7: -17.1051159, -13.4055986, -17.1532974, -13.3519802, -3.0991220, 3.1119642
8: -6.0284505, -3.2597156, -6.1186037, -3.2437406, -2.6272087, 2.7135386
9: -4.1761932, -1.8141607, -4.2196579, -1.7694638, -2.2702160, 2.2844296

Time for backsubstitution: 14.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 4636
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 6136
type: A, layer: 1, pos: 137

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 863

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6182

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2534901, upper bound: 1.2537400
time: 16.22 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2534901, upper bound: 1.2537425
time: 10.46 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -9.0225105, -5.4767828, -9.0125694, -5.4797630, -3.1541920, 3.1506524
1: -11.1949406, -7.5461597, -11.1900921, -7.5500093, -3.0061455, 2.9774537
2: -10.3165789, -6.3849535, -10.3139677, -6.3888435, -3.6104336, 3.6210885
3: -5.0169535, -2.3399386, -5.0154457, -2.3472023, -2.4105692, 2.4545820
4: -11.3779821, -8.3613138, -11.3721027, -8.3620481, -2.6059265, 2.6120648
5: 7.0747023, 9.3568459, 7.0832138, 9.3541431, -1.9212127, 1.8951468
6: -8.5178947, -5.1368351, -8.5142479, -5.1387930, -2.6588993, 2.6742549
7: -17.1522675, -13.3809681, -17.1465015, -13.3820190, -3.0908566, 3.1006918
8: -6.0411263, -3.2573314, -6.0379009, -3.2679734, -2.6201067, 2.6218498
9: -4.1997643, -1.7856164, -4.1956954, -1.7891344, -2.2780051, 2.2867165

Time for backsubstitution: 14.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 4636
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 6136
type: A, layer: 1, pos: 137

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 863

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6182

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2547427, upper bound: 1.2547420
time: 8.18 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2547427, upper bound: 1.2547418
time: 5.93 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -9.0259752, -5.4756765, -9.0468903, -5.3946738, -3.2172661, 3.1716175
1: -11.1968336, -7.5448384, -11.2442722, -7.5267091, -3.0311966, 3.0618887
2: -10.3175116, -6.3835053, -10.3301582, -6.3262587, -3.6765604, 3.6669354
3: -5.0175304, -2.3374352, -5.0540953, -2.3249912, -2.4473939, 2.4983695
4: -11.3800325, -8.3609962, -11.3931026, -8.3259354, -2.6636457, 2.6593847
5: 7.0717816, 9.3578253, 7.0602870, 9.3984661, -1.9787984, 1.9180012
6: -8.5191936, -5.1359291, -8.5433826, -5.1234598, -2.6798563, 2.7111485
7: -17.1544170, -13.3804312, -17.1745491, -13.3484173, -3.1248145, 3.1565948
8: -6.0423403, -3.2536736, -6.1233106, -3.2423029, -2.6439285, 2.7240779
9: -4.2012520, -1.7843844, -4.2232666, -1.7562397, -2.3134356, 2.3170633

Time for backsubstitution: 14.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 4636
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 6136
type: A, layer: 1, pos: 137

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 863

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6182

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2547427, upper bound: 1.2609236
time: 9.35 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2547427, upper bound: 1.2609234
time: 6.23 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -8.9979029, -5.4836698, -9.0706930, -5.3909903, -3.1784439, 3.3253248
1: -11.1782112, -7.5549293, -11.2266235, -7.5164099, -3.0280781, 3.0020561
2: -10.2911339, -6.4001079, -10.3306179, -6.3636107, -3.6231642, 3.6018739
3: -4.9516001, -2.3660431, -5.0180507, -2.3329003, -2.4022870, 2.4437027
4: -11.3666964, -8.3741102, -11.3993454, -8.3355999, -2.6255422, 2.6006932
5: 7.0910830, 9.3242636, 6.9787703, 9.3828926, -2.0401430, 1.9736809
6: -8.5002728, -5.1578631, -8.5989532, -5.0987673, -2.8499165, 2.7057016
7: -17.1029530, -13.4061375, -17.1497765, -13.3465729, -3.1056499, 3.1477933
8: -6.0272388, -3.2633767, -6.0764756, -3.2029400, -2.6692004, 2.6266069
9: -4.1746998, -1.8153970, -4.2212224, -1.7574594, -2.2804613, 2.2987833

Time for backsubstitution: 14.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 4636
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 6136
type: A, layer: 1, pos: 137

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 863

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6182

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2522780, upper bound: 1.2673423
time: 10.25 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2522780, upper bound: 1.2673421
time: 7.99 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 39.71 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 39.71
Output dim: 5, lower bound: -1.2534900, upper bound: 1.2475597
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 39.71
Output dim: 5, lower bound: -1.2534900, upper bound: 1.2475600
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 39.71
Output dim: 5, lower bound: -1.2534901, upper bound: 1.2537400
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 39.71
Output dim: 5, lower bound: -1.2534901, upper bound: 1.2537425
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 39.71
Output dim: 5, lower bound: -1.2547427, upper bound: 1.2547420
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 39.71
Output dim: 5, lower bound: -1.2547427, upper bound: 1.2547418
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 39.71
Output dim: 5, lower bound: -1.2547427, upper bound: 1.2609236
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 39.71
Output dim: 5, lower bound: -1.2547427, upper bound: 1.2609234
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 39.71
Output dim: 5, lower bound: -1.2522780, upper bound: 1.2673423
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 39.71
Output dim: 5, lower bound: -1.2522780, upper bound: 1.2673421
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 39.71
Output dim: 5, lower bound: -1.2584572, upper bound: 1.2735203
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 39.71
Output dim: 5, lower bound: -1.2597117, upper bound: 1.2745186
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 39.71
Output dim: 5, lower bound: -1.2597117, upper bound: 1.2806926
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 39.71
Output dim: 5, lower bound: -1.2673462, upper bound: 1.2584571
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 39.71
Output dim: 5, lower bound: -1.2745186, upper bound: 1.2597118
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 39.71
Output dim: 5, lower bound: -1.2735207, upper bound: 1.2584576
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 39.71
Output dim: 5, lower bound: -1.2806926, upper bound: 1.2597117
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 39.71
Output dim: 5, lower bound: -1.2463478, upper bound: 1.2644154
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 39.71
Output dim: 5, lower bound: -1.2805248, upper bound: 1.2657137
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 39.71
Output dim: 5, lower bound: -1.2794755, upper bound: 1.2644149
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 39.71
Output dim: 5, lower bound: -1.2866938, upper bound: 1.2657136
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=2.18420672416687
rel_dist={5: [-1.286786395541573, 1.2867863462497207]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5777
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 4636
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 6136
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 137

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 5777

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.1706271, upper bound: 1.1863886
time: 26.21 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1923616, upper bound: 1.1923635
time: 15.64 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 42.08 seconds
IS_A1, status: Status.VERIFIED, split count: 1, time: 42.08
Output dim: 5, lower bound: -1.1706271, upper bound: 1.1863886
IS_A2, status: Status.UNKNOWN, split count: 1, time: 42.08
Output dim: 5, lower bound: -1.1923616, upper bound: 1.1923635

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -9.0934525, -5.3854008, -9.0934582, -5.3853984, -3.3582730, 3.3593149
1: -11.2401667, -7.5092616, -11.2401724, -7.5092545, -3.0121431, 2.9965363
2: -10.3444233, -6.3544102, -10.3444290, -6.3544044, -3.6046190, 3.6175251
3: -5.0487938, -2.3199053, -5.0487976, -2.3199043, -2.4481463, 2.4480934
4: -11.4109116, -8.3298817, -11.4109154, -8.3298759, -2.5899892, 2.5810175
5: 6.9648223, 9.4015284, 6.9648027, 9.4015274, -2.0693634, 2.1325538
6: -8.6112547, -5.0921721, -8.6112642, -5.0921693, -2.8326120, 2.8638430
7: -17.1788940, -13.3413172, -17.1788979, -13.3413086, -3.1436138, 3.1427960
8: -6.0857372, -3.1872473, -6.0857406, -3.1872296, -2.6549683, 2.6256623
9: -4.2306385, -1.7395942, -4.2306414, -1.7395836, -2.3357458, 2.3183112

Time for backsubstitution: 14.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5777
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 4636
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 6136
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 137

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 5777

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.1863884, upper bound: 1.1706270
time: 11.16 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1863884, upper bound: 1.1923633
time: 11.27 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 37.48 seconds
IS_A2_B1, status: Status.VERIFIED, split count: 2, time: 37.48
Output dim: 5, lower bound: -1.1863884, upper bound: 1.1706270
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 37.48
Output dim: 5, lower bound: -1.1863884, upper bound: 1.1923633

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -9.0934525, -5.3854008, -9.0934525, -5.3854008, -3.3575182, 3.3575182
1: -11.2401667, -7.5092616, -11.2401667, -7.5092616, -2.9965267, 2.9965272
2: -10.3444233, -6.3544102, -10.3444233, -6.3544102, -3.6175194, 3.6175199
3: -5.0487938, -2.3199053, -5.0487938, -2.3199053, -2.4481440, 2.4481437
4: -11.4109116, -8.3298817, -11.4109116, -8.3298817, -2.5899820, 2.5899825
5: 6.9648223, 9.4015284, 6.9648223, 9.4015284, -2.0693631, 2.0693629
6: -8.6112547, -5.0921721, -8.6112547, -5.0921721, -2.8326097, 2.8326097
7: -17.1788940, -13.3413172, -17.1788940, -13.3413172, -3.1427937, 3.1427946
8: -6.0857372, -3.1872473, -6.0857372, -3.1872473, -2.6256590, 2.6256590
9: -4.2306385, -1.7395942, -4.2306385, -1.7395942, -2.3183093, 2.3183084

Time for backsubstitution: 14.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 4636
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 6136
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 137

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 6182

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.1813923, upper bound: 1.1765843
time: 11.34 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.1863807, upper bound: 1.1765849
time: 11.43 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 37.78 seconds
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 37.78
Output dim: 5, lower bound: -1.1813923, upper bound: 1.1765843
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 37.78
Output dim: 5, lower bound: -1.1863807, upper bound: 1.1765849
Binary search (step 2): status=Status.VERIFIED, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=2.1325697898864746
rel_dist={5: [-1.1923778842711146, 1.1923772606866638]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01171875
execution time: 1815.21 seconds
