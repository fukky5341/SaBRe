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
execution time: IAR + LP analysis = 14.82 + 38.69 = 53.51 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -1.8881189, upper bound: 1.8881190


# Binary Search by BASE starts (time budget: 3546.49 seconds, max iter: 100)

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
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 2): status=Status.UNREACHABLE, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=2.0292954444885254

## Binary Search Result
Binary search time: 182.85 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 3363.65 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5777
type: B, layer: 1, pos: 5777
type: A, layer: 1, pos: 863
type: B, layer: 1, pos: 863
type: A, layer: 1, pos: 6182
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 4636
type: B, layer: 1, pos: 4636
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 6136
type: A, layer: 1, pos: 6136
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 5777

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4209033, upper bound: 1.4523633
time: 10.62 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4581583, upper bound: 1.4581600
time: 10.31 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 21.18 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 21.18
Output dim: 5, lower bound: -1.4209033, upper bound: 1.4523633
IS_A2, status: Status.UNKNOWN, split count: 1, time: 21.18
Output dim: 5, lower bound: -1.4581583, upper bound: 1.4581600

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -9.0259924, -5.4756718, -9.0836067, -5.4139719, -3.3749237, 3.4888039
1: -11.1968470, -7.5448303, -11.2298479, -7.5201902, -3.2240524, 3.2308455
2: -10.3175230, -6.3834968, -10.3367567, -6.3618150, -3.7982149, 3.7649064
3: -5.0175495, -2.3374209, -5.0403337, -2.3251257, -2.6287446, 2.6472137
4: -11.3800459, -8.3609943, -11.4043331, -8.3388557, -2.8071308, 2.7760110
5: 7.0717616, 9.3578358, 7.0000544, 9.3988714, -2.1773028, 2.1149726
6: -8.5192099, -5.1359234, -8.5817671, -5.0960016, -3.0492764, 2.9203212
7: -17.1544342, -13.3804283, -17.1743050, -13.3540373, -3.3454647, 3.4043784
8: -6.0423503, -3.2536507, -6.0777121, -3.2087760, -2.8007450, 2.7592411
9: -4.2012644, -1.7843710, -4.2268939, -1.7531631, -2.3882208, 2.3920965

Time for backsubstitution: 14.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5777
type: A, layer: 1, pos: 863
type: B, layer: 1, pos: 863
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 444
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 4636
type: B, layer: 1, pos: 4636
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 6136
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 6136
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 5777

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4209033, upper bound: 1.4209028
time: 9.40 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4209033, upper bound: 1.4523632
time: 18.31 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -9.0934525, -5.3854008, -9.0934639, -5.3853970, -3.5902777, 3.5893631
1: -11.2401667, -7.5092616, -11.2401772, -7.5092492, -3.2806206, 3.2686839
2: -10.3444233, -6.3544102, -10.3444309, -6.3544030, -3.8091841, 3.8223758
3: -5.0487938, -2.3199053, -5.0488024, -2.3199022, -2.6788850, 2.6787779
4: -11.4109116, -8.3298817, -11.4109163, -8.3298731, -2.8254356, 2.8163133
5: 6.9648223, 9.4015284, 6.9647903, 9.4015284, -2.2332950, 2.2874773
6: -8.6112547, -5.0921721, -8.6112728, -5.0921679, -3.1187267, 3.1455159
7: -17.1788940, -13.3413172, -17.1788998, -13.3413038, -3.4497690, 3.4490647
8: -6.0857372, -3.1872473, -6.0857430, -3.1872158, -2.8326340, 2.8075073
9: -4.2306385, -1.7395942, -4.2306423, -1.7395765, -2.4413109, 2.4263618

Time for backsubstitution: 14.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5777
type: B, layer: 1, pos: 863
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 6182
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 4636
type: A, layer: 1, pos: 4636
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6136
type: B, layer: 1, pos: 6136
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 5777

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4523639, upper bound: 1.4209022
time: 17.05 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4523638, upper bound: 1.4209027
time: 12.21 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 44.34 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 44.34
Output dim: 5, lower bound: -1.4209033, upper bound: 1.4209028
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 44.34
Output dim: 5, lower bound: -1.4209033, upper bound: 1.4523632
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 44.34
Output dim: 5, lower bound: -1.4523639, upper bound: 1.4209022
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 44.34
Output dim: 5, lower bound: -1.4523638, upper bound: 1.4209027

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -9.0259924, -5.4756718, -9.0259924, -5.4756718, -3.3133888, 3.3133888
1: -11.1968470, -7.5448303, -11.1968470, -7.5448303, -3.1946936, 3.1946945
2: -10.3175230, -6.3834968, -10.3175230, -6.3834968, -3.7712288, 3.7712278
3: -5.0175495, -2.3374209, -5.0175495, -2.3374209, -2.6139493, 2.6139495
4: -11.3800459, -8.3609943, -11.3800459, -8.3609943, -2.7838683, 2.7838678
5: 7.0717616, 9.3578358, 7.0717616, 9.3578358, -2.0435972, 2.0435975
6: -8.5192099, -5.1359234, -8.5192099, -5.1359234, -2.8571711, 2.8571708
7: -17.1544342, -13.3804283, -17.1544342, -13.3804283, -3.3185701, 3.3185706
8: -6.0423503, -3.2536507, -6.0423503, -3.2536507, -2.7563591, 2.7563596
9: -4.2012644, -1.7843710, -4.2012644, -1.7843710, -2.3562622, 2.3562613

Time for backsubstitution: 14.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 863
type: B, layer: 1, pos: 863
type: A, layer: 1, pos: 6182
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 4636
type: A, layer: 1, pos: 4636
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6136
type: B, layer: 1, pos: 6136
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 863

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 863

## Relational analysis of IS_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6182

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4118979, upper bound: 1.4210859
time: 6.35 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4208764, upper bound: 1.4210833
time: 9.85 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -9.0259924, -5.4756718, -9.0934525, -5.3854008, -3.3797178, 3.4980817
1: -11.1968470, -7.5448303, -11.2401667, -7.5092616, -3.2354250, 3.2403135
2: -10.3175230, -6.3834968, -10.3444233, -6.3544102, -3.8003054, 3.7773013
3: -5.0175495, -2.3374209, -5.0487938, -2.3199053, -2.6358199, 2.6572282
4: -11.3800459, -8.3609943, -11.4109116, -8.3298817, -2.8166409, 2.7828350
5: 7.0717616, 9.3578358, 6.9648223, 9.4015284, -2.1798530, 2.1429105
6: -8.5192099, -5.1359234, -8.6112547, -5.0921721, -3.0522041, 2.9490280
7: -17.1544342, -13.3804283, -17.1788940, -13.3413172, -3.3597288, 3.4074011
8: -6.0423503, -3.2536507, -6.0857372, -3.1872473, -2.8221083, 2.7662272
9: -4.2012644, -1.7843710, -4.2306385, -1.7395942, -2.4017491, 2.3961177

Time for backsubstitution: 14.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 863
type: B, layer: 1, pos: 863
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 444
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 4636
type: B, layer: 1, pos: 4636
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 6136
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 6136
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 863

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 863

## Relational analysis of IS_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6182

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4118979, upper bound: 1.4523381
time: 6.86 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4208763, upper bound: 1.4523355
time: 13.28 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -9.0934525, -5.3854008, -9.0259924, -5.4756718, -3.4980812, 3.3797185
1: -11.2401667, -7.5092616, -11.1968470, -7.5448303, -3.2403135, 3.2354250
2: -10.3444233, -6.3544102, -10.3175230, -6.3834968, -3.7773018, 3.8003054
3: -5.0487938, -2.3199053, -5.0175495, -2.3374209, -2.6572280, 2.6358197
4: -11.4109116, -8.3298817, -11.3800459, -8.3609943, -2.7828355, 2.8166409
5: 6.9648223, 9.4015284, 7.0717616, 9.3578358, -2.1429105, 2.1798534
6: -8.6112547, -5.0921721, -8.5192099, -5.1359234, -2.9490280, 3.0522046
7: -17.1788940, -13.3413172, -17.1544342, -13.3804283, -3.4074011, 3.3597293
8: -6.0857372, -3.1872473, -6.0423503, -3.2536507, -2.7662268, 2.8221083
9: -4.2306385, -1.7395942, -4.2012644, -1.7843710, -2.3961172, 2.4017487

Time for backsubstitution: 14.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 863
type: A, layer: 1, pos: 863
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 444
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 4636
type: A, layer: 1, pos: 4636
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6136
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 6136
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 863

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4463492, upper bound: 1.3943573
time: 8.37 seconds

## Relational analysis of IS_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 863

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6182

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4523340, upper bound: 1.4118972
time: 18.44 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4523342, upper bound: 1.4208763
time: 11.46 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -9.0934525, -5.3854008, -9.0934525, -5.3854008, -3.5893545, 3.5893545
1: -11.2401667, -7.5092616, -11.2401667, -7.5092616, -3.2686682, 3.2686691
2: -10.3444233, -6.3544102, -10.3444233, -6.3544102, -3.8223648, 3.8223653
3: -5.0487938, -2.3199053, -5.0487938, -2.3199053, -2.6788788, 2.6788788
4: -11.4109116, -8.3298817, -11.4109116, -8.3298817, -2.8254232, 2.8254232
5: 6.9648223, 9.4015284, 6.9648223, 9.4015284, -2.2332931, 2.2332931
6: -8.6112547, -5.0921721, -8.6112547, -5.0921721, -3.1187234, 3.1187224
7: -17.1788940, -13.3413172, -17.1788940, -13.3413172, -3.4490604, 3.4490609
8: -6.0857372, -3.1872473, -6.0857372, -3.1872473, -2.8075018, 2.8075011
9: -4.2306385, -1.7395942, -4.2306385, -1.7395942, -2.4263587, 2.4263585

Time for backsubstitution: 14.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 863
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 6182
type: B, layer: 1, pos: 444
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 4636
type: A, layer: 1, pos: 4636
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6136
type: B, layer: 1, pos: 6136
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 863

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 863

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4463512, upper bound: 1.3943600
time: 34.22 seconds

## Relational analysis of IS_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6182

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4523361, upper bound: 1.4118979
time: 19.03 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4523361, upper bound: 1.4266175
time: 9.71 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 92.06 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 92.06
Output dim: 5, lower bound: -1.4118979, upper bound: 1.4210859
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 92.06
Output dim: 5, lower bound: -1.4208764, upper bound: 1.4210833
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 92.06
Output dim: 5, lower bound: -1.4118979, upper bound: 1.4523381
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 92.06
Output dim: 5, lower bound: -1.4208763, upper bound: 1.4523355
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 92.06
Output dim: 5, lower bound: -1.4523340, upper bound: 1.4118972
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 92.06
Output dim: 5, lower bound: -1.4523342, upper bound: 1.4208763
IS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 92.06
Output dim: 5, lower bound: -1.4523361, upper bound: 1.4118979
IS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 92.06
Output dim: 5, lower bound: -1.4523361, upper bound: 1.4266175

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -9.0125732, -5.4797626, -9.0259924, -5.4756718, -3.3115606, 3.3089252
1: -11.1900940, -7.5500093, -11.1968470, -7.5448303, -3.1544642, 3.1876960
2: -10.3139696, -6.3888435, -10.3175230, -6.3834968, -3.7610302, 3.7614102
3: -5.0154500, -2.3472006, -5.0175495, -2.3374209, -2.6105714, 2.6025445
4: -11.3721027, -8.3620472, -11.3800459, -8.3609943, -2.7742467, 2.7721381
5: 7.0832129, 9.3541451, 7.0717616, 9.3578358, -2.0317574, 2.0332453
6: -8.5142479, -5.1387920, -8.5192099, -5.1359234, -2.8512039, 2.8539772
7: -17.1465073, -13.3820181, -17.1544342, -13.3804283, -3.3095999, 3.3270855
8: -6.0379009, -3.2679739, -6.0423503, -3.2536507, -2.7464151, 2.7417450
9: -4.1956968, -1.7891331, -4.2012644, -1.7843710, -2.3597059, 2.3485608

Time for backsubstitution: 14.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 863
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 4636
type: A, layer: 1, pos: 4636
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 6136
type: A, layer: 1, pos: 6136
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 863

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 863

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 444

## Relational analysis of IS_A1_B1_A1_A1

### Relational analysis result of IS_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4112188, upper bound: 1.4111141
time: 17.35 seconds

## Relational analysis of IS_A1_B1_A1_A2

### Relational analysis result of IS_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4119885, upper bound: 1.4209763
time: 5.31 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -9.0469694, -5.3946633, -9.0259829, -5.4756751, -3.3331890, 3.3749502
1: -11.2444735, -7.5267010, -11.1968393, -7.5448375, -3.2436848, 3.2106738
2: -10.3301640, -6.3262291, -10.3175211, -6.3835011, -3.8057508, 3.8250728
3: -5.0541134, -2.3249645, -5.0175467, -2.3374300, -2.6514740, 2.6415269
4: -11.3932295, -8.3259306, -11.3800383, -8.3609962, -2.8231010, 2.8265815
5: 7.0602674, 9.3985682, 7.0717754, 9.3578329, -2.0558066, 2.0872209
6: -8.5434055, -5.1234055, -8.5192032, -5.1359272, -2.8877201, 2.8716843
7: -17.1746979, -13.3484106, -17.1544323, -13.3804293, -3.3662243, 3.3591270
8: -6.1233487, -3.2422781, -6.0423450, -3.2536664, -2.8446159, 2.7666264
9: -4.2234411, -1.7562317, -4.2012572, -1.7843757, -2.3898554, 2.3819857

Time for backsubstitution: 15.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 863
type: A, layer: 1, pos: 863
type: B, layer: 1, pos: 444
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 4636
type: A, layer: 1, pos: 4636
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6136
type: B, layer: 1, pos: 6136
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 863

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 863

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 444

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4111138, upper bound: 1.4201730
time: 7.90 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4209755, upper bound: 1.4209764
time: 10.20 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -9.0125732, -5.4797626, -9.0934525, -5.3854008, -3.3783731, 3.4935379
1: -11.1900940, -7.5500093, -11.2401667, -7.5092616, -3.1953444, 3.2333150
2: -10.3139696, -6.3888435, -10.3444233, -6.3544102, -3.7901077, 3.7683277
3: -5.0154500, -2.3472006, -5.0487938, -2.3199053, -2.6324415, 2.6456168
4: -11.3721027, -8.3620472, -11.4109116, -8.3298817, -2.8070202, 2.7707725
5: 7.0832129, 9.3541451, 6.9648223, 9.4015284, -2.1684165, 2.1317692
6: -8.5142479, -5.1387920, -8.6112547, -5.0921721, -3.0433865, 2.9443009
7: -17.1465073, -13.3820181, -17.1788940, -13.3413172, -3.3507586, 3.4006214
8: -6.0379009, -3.2679739, -6.0857372, -3.1872473, -2.8122039, 2.7523966
9: -4.1956968, -1.7891331, -4.2306385, -1.7395942, -2.4051094, 2.3890557

Time for backsubstitution: 14.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 863
type: B, layer: 1, pos: 863
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 4636
type: B, layer: 1, pos: 4636
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 6136
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 6136
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 863

## Relational analysis of IS_A1_B2_A1_A1

### Relational analysis result of IS_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.3853625, upper bound: 1.4463207
time: 9.31 seconds

## Relational analysis of IS_A1_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 863

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 444

## Relational analysis of IS_A1_B2_A1_A1

### Relational analysis result of IS_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4110120, upper bound: 1.4423999
time: 15.92 seconds

## Relational analysis of IS_A1_B2_A1_A2

### Relational analysis result of IS_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4117815, upper bound: 1.4522199
time: 5.37 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -9.0469694, -5.3946633, -9.0934448, -5.3854065, -3.3995390, 3.5386448
1: -11.2444735, -7.5267010, -11.2401609, -7.5092697, -3.2694144, 3.2562933
2: -10.3301640, -6.3262291, -10.3444176, -6.3544126, -3.8348293, 3.8316336
3: -5.0541134, -2.3249645, -5.0487905, -2.3199162, -2.6733441, 2.6854591
4: -11.3932295, -8.3259306, -11.4109039, -8.3298836, -2.8548141, 2.8264213
5: 7.0602674, 9.3985682, 6.9648337, 9.4015236, -2.1950538, 2.1543856
6: -8.5434055, -5.1234055, -8.6112490, -5.0921750, -3.0925326, 2.9592202
7: -17.1746979, -13.3484106, -17.1788883, -13.3413172, -3.4073839, 3.4491072
8: -6.1233487, -3.2422781, -6.0857320, -3.1872625, -2.8965902, 2.7791750
9: -4.2234411, -1.7562317, -4.2306328, -1.7395982, -2.4353428, 2.4225731

Time for backsubstitution: 14.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 863
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 444
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 4636
type: B, layer: 1, pos: 4636
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 6136
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 6136
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 863

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 863

## Relational analysis of IS_A1_B2_A2_A1

### Relational analysis result of IS_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.3943313, upper bound: 1.4463210
time: 7.44 seconds

## Relational analysis of IS_A1_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 444

## Relational analysis of IS_A1_B2_A2_A1

### Relational analysis result of IS_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4199659, upper bound: 1.4423997
time: 12.26 seconds

## Relational analysis of IS_A1_B2_A2_A2

### Relational analysis result of IS_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4207691, upper bound: 1.4522193
time: 10.42 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -9.0934525, -5.3854008, -9.0125732, -5.4797626, -3.4935379, 3.3783734
1: -11.2401667, -7.5092616, -11.1900940, -7.5500093, -3.2333145, 3.1953444
2: -10.3444233, -6.3544102, -10.3139696, -6.3888435, -3.7683277, 3.7901082
3: -5.0487938, -2.3199053, -5.0154500, -2.3472006, -2.6456170, 2.6324408
4: -11.4109116, -8.3298817, -11.3721027, -8.3620472, -2.7707729, 2.8070202
5: 6.9648223, 9.4015284, 7.0832129, 9.3541451, -2.1317694, 2.1684160
6: -8.6112547, -5.0921721, -8.5142479, -5.1387920, -2.9443007, 3.0433857
7: -17.1788940, -13.3413172, -17.1465073, -13.3820181, -3.4006224, 3.3507581
8: -6.0857372, -3.1872473, -6.0379009, -3.2679739, -2.7523966, 2.8122039
9: -4.2306385, -1.7395942, -4.1956968, -1.7891331, -2.3890562, 2.4051099

Time for backsubstitution: 14.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 863
type: A, layer: 1, pos: 863
type: B, layer: 1, pos: 444
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 6182
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 4636
type: A, layer: 1, pos: 4636
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6136
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 6136
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 863

## Relational analysis of IS_A2_B1_B1_B1

### Relational analysis result of IS_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4463213, upper bound: 1.3853619
time: 10.12 seconds

## Relational analysis of IS_A2_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 863

## Relational analysis of IS_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 444

## Relational analysis of IS_A2_B1_B1_B1

### Relational analysis result of IS_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4424003, upper bound: 1.4110119
time: 11.22 seconds

## Relational analysis of IS_A2_B1_B1_B2

### Relational analysis result of IS_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4522194, upper bound: 1.4117813
time: 9.91 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -9.0934448, -5.3854065, -9.0469694, -5.3946633, -3.5386438, 3.3995392
1: -11.2401609, -7.5092697, -11.2444735, -7.5267010, -3.2562933, 3.2694142
2: -10.3444176, -6.3544126, -10.3301640, -6.3262291, -3.8316336, 3.8348289
3: -5.0487905, -2.3199162, -5.0541134, -2.3249645, -2.6854587, 2.6733444
4: -11.4109039, -8.3298836, -11.3932295, -8.3259306, -2.8264217, 2.8548150
5: 6.9648337, 9.4015236, 7.0602674, 9.3985682, -2.1543856, 2.1950541
6: -8.6112490, -5.0921750, -8.5434055, -5.1234055, -2.9592199, 3.0925329
7: -17.1788883, -13.3413172, -17.1746979, -13.3484106, -3.4491072, 3.4073839
8: -6.0857320, -3.1872625, -6.1233487, -3.2422781, -2.7791748, 2.8965909
9: -4.2306328, -1.7395982, -4.2234411, -1.7562317, -2.4225731, 2.4353421

Time for backsubstitution: 14.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 863
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 444
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 6182
type: B, layer: 1, pos: 4636
type: A, layer: 1, pos: 4636
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6136
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 6136
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 863

## Relational analysis of IS_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 863

## Relational analysis of IS_A2_B1_B2_B1

### Relational analysis result of IS_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4463214, upper bound: 1.3943332
time: 8.87 seconds

## Relational analysis of IS_A2_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 444

## Relational analysis of IS_A2_B1_B2_B1

### Relational analysis result of IS_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4424003, upper bound: 1.4199658
time: 10.71 seconds

## Relational analysis of IS_A2_B1_B2_B2

### Relational analysis result of IS_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4522195, upper bound: 1.4207688
time: 10.42 seconds

## BFS IS instance: IS_A2_B2_B1

### Backsubstitution after applying IS history:
0: -9.0934525, -5.3854008, -9.0801096, -5.3896151, -3.5846767, 3.5868411
1: -11.2401667, -7.5092616, -11.2330971, -7.5143118, -3.2613726, 3.2284260
2: -10.3444233, -6.3544102, -10.3408566, -6.3596568, -3.8135557, 3.8125448
3: -5.0487938, -2.3199053, -5.0467081, -2.3295245, -2.6672254, 2.6753953
4: -11.4109116, -8.3298817, -11.4029036, -8.3309460, -2.8133245, 2.8161073
5: 6.9648223, 9.4015284, 6.9762149, 9.3975840, -2.2344880, 2.2219210
6: -8.6112547, -5.0921721, -8.6063137, -5.0953093, -3.1296773, 3.1098680
7: -17.1788940, -13.3413172, -17.1710320, -13.3429394, -3.4418068, 3.4393153
8: -6.0857372, -3.1872473, -6.0810690, -3.2015038, -2.7936649, 2.8015809
9: -4.2306385, -1.7395942, -4.2248249, -1.7442536, -2.4194527, 2.4334445

Time for backsubstitution: 14.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 863
type: B, layer: 1, pos: 863
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 444
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 4636
type: B, layer: 1, pos: 4636
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6136
type: A, layer: 1, pos: 6136
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 863

## Relational analysis of IS_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 863

## Relational analysis of IS_A2_B2_B1_B1

### Relational analysis result of IS_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4547355, upper bound: 1.4110169
time: 7.33 seconds

## Relational analysis of IS_A2_B2_B1_B2

### Relational analysis result of IS_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4581237, upper bound: 1.4175734
time: 14.43 seconds

## BFS IS instance: IS_A2_B2_B2

### Backsubstitution after applying IS history:
0: -9.0934448, -5.3854065, -9.1159449, -5.3056517, -3.6286955, 3.6101613
1: -11.2401609, -7.5092697, -11.2878428, -7.4912629, -3.2843409, 3.3209896
2: -10.3444176, -6.3544126, -10.3582697, -6.2984486, -3.8778124, 3.8568873
3: -5.0487905, -2.3199162, -5.0860376, -2.3069687, -2.7088461, 2.7172656
4: -11.4109039, -8.3298836, -11.4255095, -8.2950697, -2.8684096, 2.8592176
5: 6.9648337, 9.4015236, 6.9530535, 9.4443016, -2.2777309, 2.2491848
6: -8.6112490, -5.0921750, -8.6354847, -5.0771074, -3.1305289, 3.1600380
7: -17.1788883, -13.3413172, -17.2002563, -13.3089342, -3.4920759, 3.4797688
8: -6.0857320, -3.1872625, -6.1710987, -3.1754017, -2.8208680, 2.9029415
9: -4.2306328, -1.7395982, -4.2536182, -1.7114251, -2.4525003, 2.4592857

Time for backsubstitution: 14.91 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=2.2874808311462402
rel_dist={5: [-1.4581874533617398, 1.4581870675102389]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5777
type: B, layer: 1, pos: 5777
type: B, layer: 1, pos: 863
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 6182
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 444
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 4636
type: B, layer: 1, pos: 4636
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6136
type: B, layer: 1, pos: 6136
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 5777

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.1706271, upper bound: 1.1863886
time: 26.10 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1923616, upper bound: 1.1923635
time: 15.59 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 41.95 seconds
IS_A1, status: Status.VERIFIED, split count: 1, time: 41.95
Output dim: 5, lower bound: -1.1706271, upper bound: 1.1863886
IS_A2, status: Status.UNKNOWN, split count: 1, time: 41.95
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

Time for backsubstitution: 14.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 863
type: A, layer: 1, pos: 863
type: B, layer: 1, pos: 5777
type: A, layer: 1, pos: 6182
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 4636
type: A, layer: 1, pos: 4636
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6136
type: B, layer: 1, pos: 6136
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 863

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.1882762, upper bound: 1.1708959
time: 10.27 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1923571, upper bound: 1.1923583
time: 9.90 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 35.22 seconds
IS_A2_B1, status: Status.VERIFIED, split count: 2, time: 35.22
Output dim: 5, lower bound: -1.1882762, upper bound: 1.1708959
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 35.22
Output dim: 5, lower bound: -1.1923571, upper bound: 1.1923583

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -9.0934467, -5.3854141, -9.0934496, -5.3854170, -3.3493009, 3.3569324
1: -11.2398643, -7.5092630, -11.2396688, -7.5092573, -3.0071211, 2.9965200
2: -10.3444176, -6.3544092, -10.3444176, -6.3544092, -3.6058416, 3.6169200
3: -5.0487862, -2.3199089, -5.0487890, -2.3199067, -2.4481335, 2.4482603
4: -11.4101000, -8.3298864, -11.4095545, -8.3298807, -2.5885720, 2.5840764
5: 6.9648328, 9.4015265, 6.9648228, 9.4015255, -2.0693517, 2.1182814
6: -8.6112442, -5.0921731, -8.6112518, -5.0921707, -2.8325996, 2.8574762
7: -17.1788864, -13.3413200, -17.1788902, -13.3413124, -3.1433554, 3.1431069
8: -6.0857344, -3.1872568, -6.0857363, -3.1872439, -2.6484499, 2.6256499
9: -4.2306366, -1.7395995, -4.2306385, -1.7395926, -2.3321009, 2.3183031

Time for backsubstitution: 14.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5777
type: A, layer: 1, pos: 6182
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 444
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 863
type: B, layer: 1, pos: 4636
type: A, layer: 1, pos: 4636
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6136
type: B, layer: 1, pos: 6136
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 5777

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6182

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1873486, upper bound: 1.1923506
time: 9.75 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1923493, upper bound: 1.1923505
time: 29.26 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 60.33 seconds
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 60.33
Output dim: 5, lower bound: -1.1873486, upper bound: 1.1923506
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 60.33
Output dim: 5, lower bound: -1.1923493, upper bound: 1.1923505

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -9.0801067, -5.3896275, -9.0878525, -5.3872461, -3.3448172, 3.3570895
1: -11.2327671, -7.5143142, -11.2364950, -7.5113635, -2.9636269, 2.9537215
2: -10.3408527, -6.3596592, -10.3429298, -6.3566861, -3.5940409, 3.6040902
3: -5.0467052, -2.3295259, -5.0478702, -2.3239260, -2.4398155, 2.4350984
4: -11.4020653, -8.3309469, -11.4061651, -8.3303823, -2.5658941, 2.5665650
5: 6.9762235, 9.3975840, 6.9695830, 9.3998394, -2.0587409, 2.1121671
6: -8.6063032, -5.0953093, -8.6091604, -5.0936942, -2.8356719, 2.8612561
7: -17.1710320, -13.3429451, -17.1755543, -13.3421574, -3.1287117, 3.1333966
8: -6.0810666, -3.2015085, -6.0836935, -3.1931949, -2.6348376, 2.6067405
9: -4.2248244, -1.7442571, -4.2281294, -1.7415665, -2.3373513, 2.3240685

Time for backsubstitution: 15.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5777
type: B, layer: 1, pos: 444
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 863
type: B, layer: 1, pos: 4636
type: A, layer: 1, pos: 4636
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6136
type: B, layer: 1, pos: 6136
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 5777

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 444

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1815792, upper bound: 1.1909358
time: 7.94 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1872988, upper bound: 1.1923022
time: 20.94 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -9.1152735, -5.3057437, -9.0934343, -5.3854246, -3.3689003, 3.3939896
1: -11.2861433, -7.4913182, -11.2396593, -7.5092678, -3.0512013, 3.0121188
2: -10.3582392, -6.2986345, -10.3444109, -6.3544173, -3.6372414, 3.6720829
3: -5.0859456, -2.3071575, -5.0487852, -2.3199229, -2.4863777, 2.4729030
4: -11.4237061, -8.2951107, -11.4095392, -8.3298817, -2.6185350, 2.6270199
5: 6.9531927, 9.4436216, 6.9648428, 9.4015207, -2.0805786, 2.1367507
6: -8.6353359, -5.0774965, -8.6112394, -5.0921774, -2.8706064, 2.8688469
7: -17.1992722, -13.3089809, -17.1788788, -13.3413172, -3.1721296, 3.1860714
8: -6.1708508, -3.1755772, -6.0857286, -3.1872673, -2.7315845, 2.6347167
9: -4.2524433, -1.7114730, -4.2306290, -1.7395993, -2.3625808, 2.3443918

Time for backsubstitution: 14.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5777
type: B, layer: 1, pos: 444
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 863
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 4636
type: A, layer: 1, pos: 4636
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6136
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 6136
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 5777

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 444

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1865709, upper bound: 1.1909360
time: 12.04 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1923005, upper bound: 1.1923047
time: 11.69 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 45.25 seconds
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 45.25
Output dim: 5, lower bound: -1.1815792, upper bound: 1.1909358
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 45.25
Output dim: 5, lower bound: -1.1872988, upper bound: 1.1923022
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 45.25
Output dim: 5, lower bound: -1.1865709, upper bound: 1.1909360
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 45.25
Output dim: 5, lower bound: -1.1923005, upper bound: 1.1923047

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -9.0690126, -5.3912625, -9.0628510, -5.3942046, -3.3276629, 3.3308430
1: -11.2250948, -7.5167975, -11.2196875, -7.5202637, -2.9454265, 2.9347067
2: -10.3288679, -6.3643656, -10.3172512, -6.3720360, -3.5648746, 3.5744753
3: -5.0126395, -2.3335543, -4.9824638, -2.3501012, -2.3808069, 2.3664865
4: -11.3978357, -8.3363457, -11.3948298, -8.3432655, -2.5487690, 2.5526807
5: 6.9792643, 9.3801603, 6.9862461, 9.3672638, -2.0229816, 2.0756755
6: -8.5975647, -5.0994463, -8.5913515, -5.1152358, -2.8016057, 2.8394852
7: -17.1457596, -13.3472738, -17.1260490, -13.3675995, -3.0792694, 3.0805254
8: -6.0755949, -3.2032208, -6.0699806, -3.1992545, -2.6235371, 2.5901580
9: -4.2205253, -1.7599540, -4.2028012, -1.7714014, -2.3036966, 2.2808814

Time for backsubstitution: 14.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5777
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 4636
type: A, layer: 1, pos: 4636
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6136
type: B, layer: 1, pos: 6136
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 5777

## Relational analysis of IS_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 542

## Relational analysis of IS_A2_B2_A1_B1_B1

### Relational analysis result of IS_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1808103, upper bound: 1.1909348
time: 17.76 seconds

## Relational analysis of IS_A2_B2_A1_B1_B2

### Relational analysis result of IS_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1815780, upper bound: 1.1909348
time: 15.65 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -9.0801029, -5.3896275, -9.0878468, -5.3872476, -3.3388996, 3.3572021
1: -11.2327681, -7.5143147, -11.2364950, -7.5113645, -2.9673834, 2.9532080
2: -10.3408489, -6.3596606, -10.3429251, -6.3566890, -3.5940351, 3.5914545
3: -5.0466976, -2.3295274, -5.0478539, -2.3239288, -2.4398098, 2.3952110
4: -11.4020662, -8.3309469, -11.4061642, -8.3303833, -2.5666285, 2.5664172
5: 6.9762239, 9.3975811, 6.9695845, 9.3998318, -2.0296445, 2.1096585
6: -8.6063013, -5.0953097, -8.6091557, -5.0936952, -2.8498454, 2.8593340
7: -17.1710243, -13.3429461, -17.1755409, -13.3421574, -3.1287060, 3.1059909
8: -6.0810642, -3.2015090, -6.0836911, -3.1931973, -2.6341333, 2.6059160
9: -4.2248235, -1.7442594, -4.2281284, -1.7415725, -2.3369489, 2.3269863

Time for backsubstitution: 15.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5777
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 863
type: B, layer: 1, pos: 4636
type: A, layer: 1, pos: 4636
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6136
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 6136
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 5777

## Relational analysis of IS_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 542

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1872974, upper bound: 1.1915237
time: 36.69 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1872976, upper bound: 1.1923033
time: 20.58 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -9.1041517, -5.3073587, -9.0684280, -5.3923745, -3.3518448, 3.3677256
1: -11.2784281, -7.4937901, -11.2228603, -7.5181723, -3.0327802, 2.9930549
2: -10.3461838, -6.3032856, -10.3186884, -6.3697701, -3.6080647, 3.6424117
3: -5.0519018, -2.3111403, -4.9833813, -2.3460898, -2.4274168, 2.4043498
4: -11.4194565, -8.3004847, -11.3982029, -8.3427639, -2.6014371, 2.6131287
5: 6.9562011, 9.4261894, 6.9815016, 9.3689480, -2.0449080, 2.0937164
6: -8.6266375, -5.0816555, -8.5934420, -5.1137276, -2.8364935, 2.8469911
7: -17.1740437, -13.3132820, -17.1293831, -13.3667603, -3.1225977, 3.1332374
8: -6.1653233, -3.1772728, -6.0720100, -3.1933260, -2.7193489, 2.6179769
9: -4.2481809, -1.7271645, -4.2053051, -1.7694570, -2.3290038, 2.3012180

Time for backsubstitution: 15.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5777
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 863
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 4636
type: A, layer: 1, pos: 4636
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6136
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 6136
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 5777

## Relational analysis of IS_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 542

## Relational analysis of IS_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1858020, upper bound: 1.1909348
time: 40.23 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1865696, upper bound: 1.1909344
time: 13.05 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -9.1152706, -5.3057432, -9.0934296, -5.3854270, -3.3634615, 3.3941200
1: -11.2861414, -7.4913225, -11.2396564, -7.5092697, -3.0518365, 3.0116072
2: -10.3582373, -6.2986388, -10.3444061, -6.3544211, -3.6372347, 3.6594391
3: -5.0859385, -2.3071585, -5.0487690, -2.3199248, -2.4863720, 2.4330149
4: -11.4237051, -8.2951117, -11.4095383, -8.3298836, -2.6193147, 2.6258583
5: 6.9531932, 9.4436207, 6.9648457, 9.4015150, -2.0514817, 2.1267042
6: -8.6353321, -5.0774965, -8.6112385, -5.0921783, -2.8752942, 2.8649209
7: -17.1992645, -13.3089848, -17.1788673, -13.3413191, -3.1721239, 3.1586676
8: -6.1708508, -3.1755762, -6.0857272, -3.1872673, -2.7287369, 2.6338899
9: -4.2524433, -1.7114757, -4.2306266, -1.7396058, -2.3621788, 2.3473082

Time for backsubstitution: 15.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5777
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 863
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 4636
type: A, layer: 1, pos: 4636
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 6136
type: A, layer: 1, pos: 6136
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 5777

## Relational analysis of IS_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 542

## Relational analysis of IS_A2_B2_A2_B2_B1

### Relational analysis result of IS_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1915216, upper bound: 1.1923009
time: 25.80 seconds

## Relational analysis of IS_A2_B2_A2_B2_B2

### Relational analysis result of IS_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1922992, upper bound: 1.1923009
time: 9.22 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 56.69 seconds
IS_A2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 56.69
Output dim: 5, lower bound: -1.1808103, upper bound: 1.1909348
IS_A2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 56.69
Output dim: 5, lower bound: -1.1815780, upper bound: 1.1909348
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 56.69
Output dim: 5, lower bound: -1.1872974, upper bound: 1.1915237
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 56.69
Output dim: 5, lower bound: -1.1872976, upper bound: 1.1923033
IS_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 56.69
Output dim: 5, lower bound: -1.1858020, upper bound: 1.1909348
IS_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 56.69
Output dim: 5, lower bound: -1.1865696, upper bound: 1.1909344
IS_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 56.69
Output dim: 5, lower bound: -1.1915216, upper bound: 1.1923009
IS_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 56.69
Output dim: 5, lower bound: -1.1922992, upper bound: 1.1923009

## BFS IS instance: IS_A2_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -9.0654583, -5.3980131, -9.0495720, -5.4074836, -3.3075600, 3.3064389
1: -11.2189512, -7.5369434, -11.1884365, -7.5548606, -2.9025717, 2.8816223
2: -10.3267841, -6.3765516, -10.3069801, -6.3969564, -3.5357094, 3.5441589
3: -5.0086150, -2.3374889, -4.9710016, -2.3584447, -2.3684516, 2.3506939
4: -11.3794079, -8.3397799, -11.3597956, -8.3669662, -2.5050364, 2.5124376
5: 6.9830141, 9.3742857, 7.0005836, 9.3563175, -2.0083864, 2.0538769
6: -8.5933180, -5.1063638, -8.5793266, -5.1283770, -2.7825851, 2.8162222
7: -17.1373138, -13.3581715, -17.1083641, -13.3879824, -3.0492039, 3.0480590
8: -6.0720091, -3.2137537, -6.0549526, -3.2185965, -2.5989556, 2.5637615
9: -4.2018490, -1.7643492, -4.1701331, -1.7952356, -2.2614069, 2.2438302

Time for backsubstitution: 14.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5777
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 4636
type: A, layer: 1, pos: 4636
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6136
type: B, layer: 1, pos: 6136
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 5777

## Relational analysis of IS_A2_B2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6182

## Relational analysis of IS_A2_B2_A1_B1_B1_B1

### Relational analysis result of IS_A2_B2_A1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.1808103, upper bound: 1.1859400
time: 10.47 seconds

## Relational analysis of IS_A2_B2_A1_B1_B1_B2

### Relational analysis result of IS_A2_B2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1808103, upper bound: 1.1909348
time: 13.29 seconds

## BFS IS instance: IS_A2_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -9.0690098, -5.3912678, -9.0628471, -5.3942127, -3.3211308, 3.3272462
1: -11.2250900, -7.5168138, -11.2196846, -7.5202909, -2.9116845, 2.9344351
2: -10.3288660, -6.3643708, -10.3172483, -6.3720465, -3.5568438, 3.5689440
3: -5.0126362, -2.3335571, -4.9824572, -2.3501055, -2.3781838, 2.3664815
4: -11.3978252, -8.3363476, -11.3948107, -8.3432684, -2.5487604, 2.5349553
5: 6.9792662, 9.3801594, 6.9862471, 9.3672571, -2.0160975, 2.0720327
6: -8.5975618, -5.0994496, -8.5913506, -5.1152439, -2.7984443, 2.8366542
7: -17.1457577, -13.3472738, -17.1260395, -13.3676090, -3.0779848, 3.0847464
8: -6.0755935, -3.2032270, -6.0699778, -3.1992669, -2.6190281, 2.5901499
9: -4.2205129, -1.7599564, -4.2027798, -1.7714040, -2.3036842, 2.2615981

Time for backsubstitution: 14.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5777
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 4636
type: A, layer: 1, pos: 4636
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 6136
type: B, layer: 1, pos: 6136
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 5777

## Relational analysis of IS_A2_B2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6182

## Relational analysis of IS_A2_B2_A1_B1_B2_B1

### Relational analysis result of IS_A2_B2_A1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.1815780, upper bound: 1.1859396
time: 12.61 seconds

## Relational analysis of IS_A2_B2_A1_B1_B2_B2

### Relational analysis result of IS_A2_B2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1815780, upper bound: 1.1909348
time: 11.92 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -9.0668221, -5.4028959, -9.0842934, -5.3939924, -3.3145170, 3.3370814
1: -11.2014656, -7.5489135, -11.2303543, -7.5314956, -2.9143510, 2.9103522
2: -10.3305387, -6.3845520, -10.3407898, -6.3688722, -3.5636683, 3.5622931
3: -5.0352778, -2.3378496, -5.0438404, -2.3278565, -2.4240580, 2.3828688
4: -11.3670712, -8.3546038, -11.3877535, -8.3338089, -2.5263805, 2.5227344
5: 6.9905877, 9.3866291, 6.9733524, 9.3939581, -2.0078368, 2.0950263
6: -8.5942373, -5.1085653, -8.6049118, -5.1006351, -2.8265629, 2.8402796
7: -17.1534195, -13.3633060, -17.1671124, -13.3530579, -3.0962820, 3.0759096
8: -6.0659966, -3.2208176, -6.0800982, -3.2037234, -2.6076937, 2.5813417
9: -4.1920471, -1.7680808, -4.2094336, -1.7459838, -2.2997894, 2.2846913

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5777
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 863
type: B, layer: 1, pos: 4636
type: A, layer: 1, pos: 4636
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6136
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 6136
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 5777

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6182

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.1872976, upper bound: 1.1865186
time: 14.43 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1872974, upper bound: 1.1915237
time: 26.24 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -9.0801020, -5.3896346, -9.0878439, -5.3872514, -3.3352823, 3.3506544
1: -11.2327604, -7.5143404, -11.2364893, -7.5113802, -2.9641004, 2.9194670
2: -10.3408480, -6.3596716, -10.3429222, -6.3566928, -3.5884972, 3.5834994
3: -5.0466971, -2.3295298, -5.0478539, -2.3239319, -2.4398046, 2.3925869
4: -11.4020443, -8.3309488, -11.4061508, -8.3303833, -2.5489011, 2.5664082
5: 6.9762268, 9.3975763, 6.9695873, 9.3998308, -2.0281272, 2.1023147
6: -8.6063004, -5.0953164, -8.6091528, -5.0936995, -2.8470421, 2.8561234
7: -17.1710205, -13.3429527, -17.1755371, -13.3421631, -3.1329250, 3.1047058
8: -6.0810633, -3.2015190, -6.0836906, -3.1932025, -2.6341257, 2.6014082
9: -4.2248025, -1.7442625, -4.2281160, -1.7415740, -2.3176656, 2.3269739

Time for backsubstitution: 14.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5777
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 863
type: B, layer: 1, pos: 4636
type: A, layer: 1, pos: 4636
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 6136
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 6136
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 5777

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6182

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.1872976, upper bound: 1.1872965
time: 24.83 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1872976, upper bound: 1.1923033
time: 15.69 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -9.1005630, -5.3141232, -9.0551414, -5.4056740, -3.3316875, 3.3432300
1: -11.2722597, -7.5139465, -11.1915512, -7.5527725, -2.9898024, 2.9399838
2: -10.3440437, -6.3155355, -10.3083830, -6.3947229, -3.5788565, 3.6121283
3: -5.0478764, -2.3150709, -4.9719057, -2.3544331, -2.4150400, 2.3885288
4: -11.4010305, -8.3039265, -11.3631687, -8.3664789, -2.5574594, 2.5728188
5: 6.9599910, 9.4202881, 6.9958467, 9.3579893, -2.0303411, 2.0712361
6: -8.6224432, -5.0886059, -8.5814056, -5.1269350, -2.8174076, 2.8237081
7: -17.1655941, -13.3242054, -17.1116982, -13.3871899, -3.0918150, 3.1004238
8: -6.1617622, -3.1878004, -6.0569592, -3.2126732, -2.6948280, 2.5915997
9: -4.2294979, -1.7315602, -4.1726136, -1.7933009, -2.2865753, 2.2641277

Time for backsubstitution: 14.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5777
type: A, layer: 1, pos: 863
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 4636
type: A, layer: 1, pos: 4636
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6136
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 6136
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 5777

## Relational analysis of IS_A2_B2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.
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

## Binary Search with IS_dual Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 1720.06 seconds
