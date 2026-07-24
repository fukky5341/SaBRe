## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.5245661088


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-16.8120441, -14.7792912, -16.8120441, -14.7792912, -1.0760951, 1.0760951)
1: (-16.2553310, -14.1722717, -16.2553310, -14.1722717, -1.4490080, 1.4490080)
2: (-12.0810032, -10.6060753, -12.0810032, -10.6060753, -1.0568600, 1.0568600)
3: (-11.9948082, -10.3719950, -11.9948082, -10.3719950, -1.3407326, 1.3407326)
4: (-2.2575240, -1.1358814, -2.2575240, -1.1358814, -0.9267054, 0.9267054)
5: (-8.1275635, -6.6195669, -8.1275635, -6.6195669, -0.8810911, 0.8810914)
6: (-16.8353367, -15.0711985, -16.8353367, -15.0711985, -0.9718437, 0.9718435)
7: (-6.8141041, -5.2119651, -6.8141041, -5.2119651, -1.2129564, 1.2129569)
8: (-3.6343689, -2.3530221, -3.6343689, -2.3530221, -1.2451773, 1.2451777)
9: (5.4975553, 6.7105274, 5.4975553, 6.7105274, -0.9138064, 0.9138067)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.98 + 35.11 = 58.08 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.5250911, upper bound: 0.5250914

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 5759
type: A, layer: 1, pos: 5759
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 6236
type: B, layer: 1, pos: 6236
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 145
type: A, layer: 1, pos: 145
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 845

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5248750, upper bound: 0.5247637
time: 4.76 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5247633, upper bound: 0.5247637
time: 4.96 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 9.96 seconds
NS_B1, status: Status.UNKNOWN, split count: 1, time: 9.96
Output dim: 9, lower bound: -0.5248750, upper bound: 0.5247637
NS_B2, status: Status.UNKNOWN, split count: 1, time: 9.96
Output dim: 9, lower bound: -0.5247633, upper bound: 0.5247637

## BFS NS instance: NS_B1

### Backsubstitution after applying NS history:
0: -16.8119717, -14.7808495, -16.8118744, -14.7827873, -1.0719867, 1.0741301
1: -16.2532425, -14.1725426, -16.2506256, -14.1728897, -1.4461145, 1.4433947
2: -12.0797310, -10.6063452, -12.0781336, -10.6066895, -1.0551620, 1.0544605
3: -11.9946671, -10.3721228, -11.9944878, -10.3722916, -1.3400259, 1.3400803
4: -2.2573123, -1.1360272, -2.2570481, -1.1362169, -0.9246087, 0.9256430
5: -8.1274948, -6.6219063, -8.1274128, -6.6248302, -0.8752856, 0.8783855
6: -16.8348351, -15.0712833, -16.8342171, -15.0713892, -0.9708509, 0.9699345
7: -6.8139114, -5.2144313, -6.8136673, -5.2175102, -1.2073922, 1.2103243
8: -3.6339922, -2.3533592, -3.6335135, -2.3537788, -1.2435360, 1.2437844
9: 5.4977684, 6.7104955, 5.4980354, 6.7104521, -0.9129844, 0.9121387

Time for backsubstitution: 21.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 5759
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 5759
type: B, layer: 1, pos: 6236
type: A, layer: 1, pos: 6236
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 145
type: B, layer: 1, pos: 145
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 845

## Relational analysis of NS_B1_A1

### Relational analysis result of NS_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5246738, upper bound: 0.5246752
time: 3.68 seconds

## Relational analysis of NS_B1_A2

### Relational analysis result of NS_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5246738, upper bound: 0.5247648
time: 3.79 seconds

## BFS NS instance: NS_B2

### Backsubstitution after applying NS history:
0: -16.8120461, -14.7792988, -16.8185768, -14.7792740, -1.0743632, 1.0826459
1: -16.2553291, -14.1722746, -16.2566032, -14.1614799, -1.4599400, 1.4470320
2: -12.0809984, -10.6060772, -12.0813856, -10.5969620, -1.0673470, 1.0573564
3: -11.9947958, -10.3720665, -11.9946308, -10.3698521, -1.3434696, 1.3458934
4: -2.2572832, -1.1358852, -2.2577202, -1.1360559, -0.9228177, 0.9317465
5: -8.1275625, -6.6195807, -8.1365108, -6.6192932, -0.8776813, 0.8887172
6: -16.8353291, -15.0712013, -16.8356094, -15.0689297, -0.9740710, 0.9704192
7: -6.8141031, -5.2119765, -6.8237934, -5.2111654, -1.2118320, 1.2228208
8: -3.6343646, -2.3530240, -3.6397457, -2.3528509, -1.2462254, 1.2516904
9: 5.4975581, 6.7103491, 5.4964356, 6.7102175, -0.9177604, 0.9121141

Time for backsubstitution: 21.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 5759
type: A, layer: 1, pos: 5759
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 6236
type: A, layer: 1, pos: 6236
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 145
type: B, layer: 1, pos: 145
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 845

## Relational analysis of NS_B2_A1

### Relational analysis result of NS_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5247634, upper bound: 0.5246741
time: 4.43 seconds

## Relational analysis of NS_B2_A2

### Relational analysis result of NS_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5247634, upper bound: 0.5247637
time: 4.36 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 30.79 seconds
NS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 30.79
Output dim: 9, lower bound: -0.5246738, upper bound: 0.5246752
NS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 30.79
Output dim: 9, lower bound: -0.5246738, upper bound: 0.5247648
NS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 30.79
Output dim: 9, lower bound: -0.5247634, upper bound: 0.5246741
NS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 30.79
Output dim: 9, lower bound: -0.5247634, upper bound: 0.5247637

## BFS NS instance: NS_B1_A1

### Backsubstitution after applying NS history:
0: -16.8118744, -14.7827873, -16.8118744, -14.7827873, -1.0718894, 1.0718894
1: -16.2506256, -14.1728897, -16.2506256, -14.1728897, -1.4431162, 1.4431167
2: -12.0781336, -10.6066895, -12.0781336, -10.6066895, -1.0541720, 1.0541725
3: -11.9944878, -10.3722916, -11.9944878, -10.3722916, -1.3397884, 1.3397889
4: -2.2570481, -1.1362169, -2.2570481, -1.1362169, -0.9245191, 0.9245191
5: -8.1274128, -6.6248302, -8.1274128, -6.6248302, -0.8751969, 0.8751965
6: -16.8342171, -15.0713892, -16.8342171, -15.0713892, -0.9698319, 0.9698319
7: -6.8136673, -5.2175102, -6.8136673, -5.2175102, -1.2072835, 1.2072845
8: -3.6335135, -2.3537788, -3.6335135, -2.3537788, -1.2430696, 1.2430701
9: 5.4980354, 6.7104521, 5.4980354, 6.7104521, -0.9120822, 0.9120822

Time for backsubstitution: 22.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5759
type: A, layer: 1, pos: 5759
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 6236
type: B, layer: 1, pos: 6236
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 145
type: A, layer: 1, pos: 145
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 1, pos: 5759

## Relational analysis of NS_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5759

### Candidate
type: B, layer: 1, pos: 524

## Relational analysis of NS_B1_A1_B1

### Relational analysis result of NS_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5248742, upper bound: 0.5246265
time: 4.03 seconds

## Relational analysis of NS_B1_A1_B2

### Relational analysis result of NS_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5248742, upper bound: 0.5246743
time: 3.88 seconds

## BFS NS instance: NS_B1_A2

### Backsubstitution after applying NS history:
0: -16.8185768, -14.7792740, -16.8118744, -14.7827873, -1.0786176, 1.0753584
1: -16.2566032, -14.1614799, -16.2506256, -14.1728897, -1.4492140, 1.4545531
2: -12.0813856, -10.5969620, -12.0781336, -10.6066895, -1.0572658, 1.0651727
3: -11.9946308, -10.3698521, -11.9944878, -10.3722916, -1.3438892, 1.3438091
4: -2.2577202, -1.1360559, -2.2570481, -1.1362169, -0.9249396, 0.9245720
5: -8.1365108, -6.6192932, -8.1274128, -6.6248302, -0.8829832, 0.8807912
6: -16.8356094, -15.0689297, -16.8342171, -15.0713892, -0.9712958, 0.9722526
7: -6.8237934, -5.2111654, -6.8136673, -5.2175102, -1.2173524, 1.2137423
8: -3.6397457, -2.3528509, -3.6335135, -2.3537788, -1.2504168, 1.2445664
9: 5.4964356, 6.7102175, 5.4980354, 6.7104521, -0.9136357, 0.9120793

Time for backsubstitution: 21.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 5759
type: B, layer: 1, pos: 5759
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 6236
type: B, layer: 1, pos: 6236
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 145
type: A, layer: 1, pos: 145
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 524

## Relational analysis of NS_B1_A2_A1

### Relational analysis result of NS_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5248262, upper bound: 0.5247639
time: 3.60 seconds

## Relational analysis of NS_B1_A2_A2

### Relational analysis result of NS_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5248741, upper bound: 0.5247628
time: 4.68 seconds

## BFS NS instance: NS_B2_A1

### Backsubstitution after applying NS history:
0: -16.8118744, -14.7827873, -16.8185768, -14.7792740, -1.0753579, 1.0786176
1: -16.2506256, -14.1728897, -16.2566032, -14.1614799, -1.4545536, 1.4492149
2: -12.0781336, -10.6066895, -12.0813856, -10.5969620, -1.0651722, 1.0572658
3: -11.9944878, -10.3722916, -11.9946308, -10.3698521, -1.3438091, 1.3438892
4: -2.2570481, -1.1362169, -2.2577202, -1.1360559, -0.9245720, 0.9249396
5: -8.1274128, -6.6248302, -8.1365108, -6.6192932, -0.8807912, 0.8829830
6: -16.8342171, -15.0713892, -16.8356094, -15.0689297, -0.9722528, 0.9712958
7: -6.8136673, -5.2175102, -6.8237934, -5.2111654, -1.2137423, 1.2173524
8: -3.6335135, -2.3537788, -3.6397457, -2.3528509, -1.2445669, 1.2504168
9: 5.4980354, 6.7104521, 5.4964356, 6.7102175, -0.9120793, 0.9136357

Time for backsubstitution: 22.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 5759
type: A, layer: 1, pos: 5759
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 6236
type: A, layer: 1, pos: 6236
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 145
type: B, layer: 1, pos: 145
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 524

## Relational analysis of NS_B2_A1_B1

### Relational analysis result of NS_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5246730, upper bound: 0.5246265
time: 3.73 seconds

## Relational analysis of NS_B2_A1_B2

### Relational analysis result of NS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5246730, upper bound: 0.5246738
time: 3.68 seconds

## BFS NS instance: NS_B2_A2

### Backsubstitution after applying NS history:
0: -16.8185768, -14.7792740, -16.8185768, -14.7792740, -1.0748701, 1.0748701
1: -16.2566032, -14.1614799, -16.2566032, -14.1614799, -1.4503336, 1.4503331
2: -12.0813856, -10.5969620, -12.0813856, -10.5969620, -1.0646291, 1.0646291
3: -11.9946308, -10.3698521, -11.9946308, -10.3698521, -1.3483124, 1.3483114
4: -2.2577202, -1.1360559, -2.2577202, -1.1360559, -0.9321232, 0.9321237
5: -8.1365108, -6.6192932, -8.1365108, -6.6192932, -0.8778796, 0.8778799
6: -16.8356094, -15.0689297, -16.8356094, -15.0689297, -0.9705925, 0.9705925
7: -6.8237934, -5.2111654, -6.8237934, -5.2111654, -1.2157402, 1.2157402
8: -3.6397457, -2.3528509, -3.6397457, -2.3528509, -1.2513485, 1.2513485
9: 5.4964356, 6.7102175, 5.4964356, 6.7102175, -0.9186363, 0.9186363

Time for backsubstitution: 22.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5759
type: B, layer: 1, pos: 5759
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 6236
type: A, layer: 1, pos: 6236
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 145
type: B, layer: 1, pos: 145
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 5759

## Relational analysis of NS_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5759

### Candidate
type: B, layer: 1, pos: 524

## Relational analysis of NS_B2_A2_B1

### Relational analysis result of NS_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5246730, upper bound: 0.5247162
time: 3.50 seconds

## Relational analysis of NS_B2_A2_B2

### Relational analysis result of NS_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5246730, upper bound: 0.5247639
time: 3.65 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 35.19 seconds
NS_B1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 35.19
Output dim: 9, lower bound: -0.5248742, upper bound: 0.5246265
NS_B1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 35.19
Output dim: 9, lower bound: -0.5248742, upper bound: 0.5246743
NS_B1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 35.19
Output dim: 9, lower bound: -0.5248262, upper bound: 0.5247639
NS_B1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 35.19
Output dim: 9, lower bound: -0.5248741, upper bound: 0.5247628
NS_B2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 35.19
Output dim: 9, lower bound: -0.5246730, upper bound: 0.5246265
NS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 35.19
Output dim: 9, lower bound: -0.5246730, upper bound: 0.5246738
NS_B2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 35.19
Output dim: 9, lower bound: -0.5246730, upper bound: 0.5247162
NS_B2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 35.19
Output dim: 9, lower bound: -0.5246730, upper bound: 0.5247639

## BFS NS instance: NS_B1_A1_B1

### Backsubstitution after applying NS history:
0: -16.8118744, -14.7827873, -16.8117599, -14.7829723, -1.0711384, 1.0711722
1: -16.2506256, -14.1728897, -16.2504559, -14.1760826, -1.4398894, 1.4429235
2: -12.0781336, -10.6066895, -12.0781193, -10.6079550, -1.0527449, 1.0541425
3: -11.9944878, -10.3722916, -11.9937553, -10.3724556, -1.3395510, 1.3389525
4: -2.2570481, -1.1362169, -2.2567663, -1.1375749, -0.9231544, 0.9243474
5: -8.1274128, -6.6248302, -8.1264286, -6.6248889, -0.8751426, 0.8741724
6: -16.8342171, -15.0713892, -16.8341293, -15.0735855, -0.9675159, 0.9694788
7: -6.8136673, -5.2175102, -6.8134255, -5.2176728, -1.2062778, 1.2062750
8: -3.6335135, -2.3537788, -3.6332235, -2.3546948, -1.2400913, 1.2410040
9: 5.4980354, 6.7104521, 5.4982548, 6.7104034, -0.9119825, 0.9117608

Time for backsubstitution: 21.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 5759
type: B, layer: 1, pos: 5759
type: A, layer: 1, pos: 6236
type: B, layer: 1, pos: 6236
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 145
type: A, layer: 1, pos: 145
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 524

## Relational analysis of NS_B1_A1_B1_A1

### Relational analysis result of NS_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5248900, upper bound: 0.5248901
time: 6.62 seconds

## Relational analysis of NS_B1_A1_B1_A2

### Relational analysis result of NS_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5248900, upper bound: 0.5248899
time: 7.31 seconds

## BFS NS instance: NS_B1_A1_B2

### Backsubstitution after applying NS history:
0: -16.8118763, -14.7827902, -16.8188763, -14.7826519, -1.0845814, 1.0764256
1: -16.2506275, -14.1729040, -16.3226395, -14.1662626, -1.4523678, 1.4628830
2: -12.0781336, -10.6066971, -12.1072292, -10.5934496, -1.0616393, 1.0672908
3: -11.9944849, -10.3722935, -11.9998159, -10.3451881, -1.3552890, 1.3462849
4: -2.2570477, -1.1362216, -2.2835135, -1.1341355, -0.9273176, 0.9403346
5: -8.1274099, -6.6248312, -8.1314659, -6.6055527, -0.8831091, 0.8785532
6: -16.8342190, -15.0713978, -16.8826714, -15.0701351, -0.9785309, 0.9837971
7: -6.8136659, -5.2175117, -6.8180022, -5.2145395, -1.2089911, 1.2305574
8: -3.6335130, -2.3537817, -3.6812420, -2.3537087, -1.2620802, 1.2579331
9: 5.4980359, 6.7104526, 5.4961758, 6.7154632, -0.9168630, 0.9158747

Time for backsubstitution: 21.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5759
type: B, layer: 1, pos: 5759
type: A, layer: 1, pos: 6236
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 6236
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 145
type: A, layer: 1, pos: 145
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 5759

### Candidate
type: B, layer: 1, pos: 5759

### Candidate
type: A, layer: 1, pos: 6236

## Relational analysis of NS_B1_A1_B2_A1

### Relational analysis result of NS_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5242204, upper bound: 0.5246899
time: 5.33 seconds

## Relational analysis of NS_B1_A1_B2_A2

### Relational analysis result of NS_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5249389, upper bound: 0.5249403
time: 4.05 seconds

## BFS NS instance: NS_B1_A2_A1

### Backsubstitution after applying NS history:
0: -16.8184566, -14.7794638, -16.8118744, -14.7827873, -1.0778980, 1.0746064
1: -16.2564297, -14.1646700, -16.2506256, -14.1728897, -1.4490194, 1.4513268
2: -12.0813694, -10.5982246, -12.0781336, -10.6066895, -1.0572348, 1.0637479
3: -11.9938993, -10.3700199, -11.9944878, -10.3722916, -1.3430519, 1.3435693
4: -2.2574382, -1.1374135, -2.2570481, -1.1362169, -0.9247684, 0.9232073
5: -8.1355286, -6.6193523, -8.1274128, -6.6248302, -0.8819580, 0.8807368
6: -16.8355217, -15.0711288, -16.8342171, -15.0713892, -0.9709516, 0.9699376
7: -6.8235569, -5.2113314, -6.8136673, -5.2175102, -1.2163463, 1.2127371
8: -3.6394529, -2.3537650, -3.6335135, -2.3537788, -1.2483521, 1.2415881
9: 5.4966555, 6.7101684, 5.4980354, 6.7104521, -0.9133153, 0.9119802

Time for backsubstitution: 21.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5759
type: A, layer: 1, pos: 5759
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 6236
type: A, layer: 1, pos: 6236
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 145
type: A, layer: 1, pos: 145
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 5759

## Relational analysis of NS_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5759

### Candidate
type: B, layer: 1, pos: 524

## Relational analysis of NS_B1_A2_A1_B1

### Relational analysis result of NS_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5248259, upper bound: 0.5247157
time: 3.62 seconds

## Relational analysis of NS_B1_A2_A1_B2

### Relational analysis result of NS_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5248259, upper bound: 0.5247637
time: 3.67 seconds

## BFS NS instance: NS_B1_A2_A2

### Backsubstitution after applying NS history:
0: -16.8255730, -14.7791367, -16.8118763, -14.7827902, -1.0831542, 1.0861666
1: -16.3286018, -14.1548710, -16.2506275, -14.1729040, -1.4666061, 1.4637976
2: -12.1104774, -10.5837841, -12.0781336, -10.6066971, -1.0700488, 1.0725985
3: -12.0000219, -10.3426847, -11.9944849, -10.3722935, -1.3503370, 1.3583612
4: -2.2841709, -1.1339774, -2.2570477, -1.1362216, -0.9413722, 0.9273720
5: -8.1405516, -6.6000161, -8.1274099, -6.6248312, -0.8858280, 0.8859143
6: -16.8840809, -15.0676727, -16.8342190, -15.0713978, -0.9849377, 0.9809501
7: -6.8281164, -5.2081923, -6.8136659, -5.2175117, -1.2331972, 1.2154555
8: -3.6875019, -2.3527803, -3.6335130, -2.3537817, -1.2618427, 1.2636042
9: 5.4945602, 6.7152224, 5.4980359, 6.7104526, -0.9174519, 0.9168568

Time for backsubstitution: 21.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5759
type: A, layer: 1, pos: 5759
type: B, layer: 1, pos: 6236
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 6236
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 145
type: B, layer: 1, pos: 145
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 5759

## Relational analysis of NS_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5759

### Candidate
type: B, layer: 1, pos: 6236

## Relational analysis of NS_B1_A2_A2_B1

### Relational analysis result of NS_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5246218, upper bound: 0.5240420
time: 4.61 seconds

## Relational analysis of NS_B1_A2_A2_B2

### Relational analysis result of NS_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5248742, upper bound: 0.5247637
time: 3.92 seconds

## BFS NS instance: NS_B2_A1_B1

### Backsubstitution after applying NS history:
0: -16.8118744, -14.7827873, -16.8184566, -14.7794638, -1.0746069, 1.0778980
1: -16.2506256, -14.1728897, -16.2564297, -14.1646700, -1.4513273, 1.4490204
2: -12.0781336, -10.6066895, -12.0813694, -10.5982246, -1.0637479, 1.0572348
3: -11.9944878, -10.3722916, -11.9938993, -10.3700199, -1.3435688, 1.3430519
4: -2.2570481, -1.1362169, -2.2574382, -1.1374135, -0.9232073, 0.9247684
5: -8.1274128, -6.6248302, -8.1355286, -6.6193523, -0.8807368, 0.8819578
6: -16.8342171, -15.0713892, -16.8355217, -15.0711288, -0.9699373, 0.9709516
7: -6.8136673, -5.2175102, -6.8235569, -5.2113314, -1.2127371, 1.2163467
8: -3.6335135, -2.3537788, -3.6394529, -2.3537650, -1.2415886, 1.2483530
9: 5.4980354, 6.7104521, 5.4966555, 6.7101684, -0.9119802, 0.9133155

Time for backsubstitution: 21.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5759
type: B, layer: 1, pos: 5759
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 6236
type: B, layer: 1, pos: 6236
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 145
type: B, layer: 1, pos: 145
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 5759

## Relational analysis of NS_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5759

### Candidate
type: A, layer: 1, pos: 524

## Relational analysis of NS_B2_A1_B1_A1

### Relational analysis result of NS_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5247143, upper bound: 0.5248273
time: 3.60 seconds

## Relational analysis of NS_B2_A1_B1_A2

### Relational analysis result of NS_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5247143, upper bound: 0.5248273
time: 3.69 seconds

## BFS NS instance: NS_B2_A1_B2

### Backsubstitution after applying NS history:
0: -16.8118763, -14.7827902, -16.8255730, -14.7791367, -1.0861669, 1.0831542
1: -16.2506275, -14.1729040, -16.3286018, -14.1548710, -1.4637971, 1.4666061
2: -12.0781336, -10.6066971, -12.1104774, -10.5837841, -1.0725985, 1.0700488
3: -11.9944849, -10.3722935, -12.0000219, -10.3426847, -1.3583608, 1.3503375
4: -2.2570477, -1.1362216, -2.2841709, -1.1339774, -0.9273720, 0.9413722
5: -8.1274099, -6.6248312, -8.1405516, -6.6000161, -0.8859143, 0.8858278
6: -16.8342190, -15.0713978, -16.8840809, -15.0676727, -0.9809504, 0.9849374
7: -6.8136659, -5.2175117, -6.8281164, -5.2081923, -1.2154560, 1.2331972
8: -3.6335130, -2.3537817, -3.6875019, -2.3527803, -1.2636042, 1.2618432
9: 5.4980359, 6.7104526, 5.4945602, 6.7152224, -0.9168568, 0.9174516

Time for backsubstitution: 21.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5759
type: B, layer: 1, pos: 5759
type: A, layer: 1, pos: 6236
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 6236
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 145
type: A, layer: 1, pos: 145
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 5759

## Relational analysis of NS_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5759

### Candidate
type: A, layer: 1, pos: 6236

## Relational analysis of NS_B2_A1_B2_A1

### Relational analysis result of NS_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5240417, upper bound: 0.5246219
time: 4.71 seconds

## Relational analysis of NS_B2_A1_B2_A2

### Relational analysis result of NS_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5247623, upper bound: 0.5248745
time: 4.63 seconds

## BFS NS instance: NS_B2_A2_B1

### Backsubstitution after applying NS history:
0: -16.8185768, -14.7792740, -16.8184566, -14.7794638, -1.0741210, 1.0741520
1: -16.2566032, -14.1614799, -16.2564297, -14.1646700, -1.4471059, 1.4501371
2: -12.0813856, -10.5969620, -12.0813694, -10.5982246, -1.0632043, 1.0645971
3: -11.9946308, -10.3698521, -11.9938993, -10.3700199, -1.3480711, 1.3474751
4: -2.2577202, -1.1360559, -2.2574382, -1.1374135, -0.9307599, 0.9319530
5: -8.1365108, -6.6192932, -8.1355286, -6.6193523, -0.8778253, 0.8768559
6: -16.8356094, -15.0689297, -16.8355217, -15.0711288, -0.9682779, 0.9702482
7: -6.8237934, -5.2111654, -6.8235569, -5.2113314, -1.2147346, 1.2147336
8: -3.6397457, -2.3528509, -3.6394529, -2.3537650, -1.2483692, 1.2492852
9: 5.4964356, 6.7102175, 5.4966555, 6.7101684, -0.9185367, 0.9183154

Time for backsubstitution: 22.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 5759
type: B, layer: 1, pos: 5759
type: A, layer: 1, pos: 6236
type: B, layer: 1, pos: 6236
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 145
type: A, layer: 1, pos: 145
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 524

## Relational analysis of NS_B2_A2_B1_A1

### Relational analysis result of NS_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5246249, upper bound: 0.5247158
time: 3.73 seconds

## Relational analysis of NS_B2_A2_B1_A2

### Relational analysis result of NS_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5246249, upper bound: 0.5247158
time: 3.57 seconds

## BFS NS instance: NS_B2_A2_B2

### Backsubstitution after applying NS history:
0: -16.8185730, -14.7792721, -16.8255730, -14.7791367, -1.0887952, 1.0794086
1: -16.2566013, -14.1614914, -16.3286018, -14.1548710, -1.4595766, 1.4722471
2: -12.0813847, -10.5969667, -12.1104774, -10.5837841, -1.0720544, 1.0752897
3: -11.9946289, -10.3698549, -12.0000219, -10.3426847, -1.3612823, 1.3547597
4: -2.2577209, -1.1360621, -2.2841709, -1.1339774, -0.9349241, 0.9459851
5: -8.1365089, -6.6192956, -8.1405516, -6.6000161, -0.8886867, 0.8812330
6: -16.8356113, -15.0689383, -16.8840809, -15.0676727, -0.9792910, 0.9870009
7: -6.8237944, -5.2111692, -6.8281164, -5.2081923, -1.2174549, 1.2375679
8: -3.6397452, -2.3528552, -3.6875019, -2.3527803, -1.2683630, 1.2637444
9: 5.4964361, 6.7102151, 5.4945602, 6.7152224, -0.9234152, 0.9224329

Time for backsubstitution: 21.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5759
type: B, layer: 1, pos: 5759
type: A, layer: 1, pos: 6236
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 6236
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 145
type: A, layer: 1, pos: 145
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 5759

### Candidate
type: B, layer: 1, pos: 5759

### Candidate
type: A, layer: 1, pos: 6236

## Relational analysis of NS_B2_A2_B2_A1

### Relational analysis result of NS_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.5239532, upper bound: 0.5245126
time: 3.82 seconds

## Relational analysis of NS_B2_A2_B2_A2

### Relational analysis result of NS_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5246729, upper bound: 0.5247639
time: 3.91 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 29.98 seconds
NS_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 29.98
Output dim: 9, lower bound: -0.5248900, upper bound: 0.5248901
NS_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 29.98
Output dim: 9, lower bound: -0.5248900, upper bound: 0.5248899
NS_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 29.98
Output dim: 9, lower bound: -0.5242204, upper bound: 0.5246899
NS_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 29.98
Output dim: 9, lower bound: -0.5249389, upper bound: 0.5249403
NS_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 29.98
Output dim: 9, lower bound: -0.5248259, upper bound: 0.5247157
NS_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 29.98
Output dim: 9, lower bound: -0.5248259, upper bound: 0.5247637
NS_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 29.98
Output dim: 9, lower bound: -0.5246218, upper bound: 0.5240420
NS_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 29.98
Output dim: 9, lower bound: -0.5248742, upper bound: 0.5247637
NS_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 29.98
Output dim: 9, lower bound: -0.5247143, upper bound: 0.5248273
NS_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 29.98
Output dim: 9, lower bound: -0.5247143, upper bound: 0.5248273
NS_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 29.98
Output dim: 9, lower bound: -0.5240417, upper bound: 0.5246219
NS_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 29.98
Output dim: 9, lower bound: -0.5247623, upper bound: 0.5248745
NS_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 29.98
Output dim: 9, lower bound: -0.5246249, upper bound: 0.5247158
NS_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 29.98
Output dim: 9, lower bound: -0.5246249, upper bound: 0.5247158
NS_B2_A2_B2_A1, status: Status.VERIFIED, split count: 4, time: 29.98
Output dim: 9, lower bound: -0.5239532, upper bound: 0.5245126
NS_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 29.98
Output dim: 9, lower bound: -0.5246729, upper bound: 0.5247639

## BFS NS instance: NS_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -16.8117599, -14.7829723, -16.8117599, -14.7829723, -1.0704212, 1.0704217
1: -16.2504559, -14.1760826, -16.2504559, -14.1760826, -1.4396958, 1.4396963
2: -12.0781193, -10.6079550, -12.0781193, -10.6079550, -1.0527153, 1.0527153
3: -11.9937553, -10.3724556, -11.9937553, -10.3724556, -1.3387156, 1.3387151
4: -2.2567663, -1.1375749, -2.2567663, -1.1375749, -0.9229827, 0.9229827
5: -8.1264286, -6.6248889, -8.1264286, -6.6248889, -0.8741183, 0.8741186
6: -16.8341293, -15.0735855, -16.8341293, -15.0735855, -0.9671631, 0.9671631
7: -6.8134255, -5.2176728, -6.8134255, -5.2176728, -1.2052689, 1.2052684
8: -3.6332235, -2.3546948, -3.6332235, -2.3546948, -1.2380247, 1.2380252
9: 5.4982548, 6.7104034, 5.4982548, 6.7104034, -0.9116611, 0.9116614

Time for backsubstitution: 22.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5759
type: B, layer: 1, pos: 5759
type: B, layer: 1, pos: 6236
type: A, layer: 1, pos: 6236
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 145
type: A, layer: 1, pos: 145
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 1, pos: 5759

### Candidate
type: B, layer: 1, pos: 5759

### Candidate
type: B, layer: 1, pos: 6236

## Relational analysis of NS_B1_A1_B1_A1_B1

### Relational analysis result of NS_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5246415, upper bound: 0.5241727
time: 5.79 seconds

## Relational analysis of NS_B1_A1_B1_A1_B2

### Relational analysis result of NS_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5248900, upper bound: 0.5248900
time: 5.59 seconds

## BFS NS instance: NS_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -16.8188763, -14.7826519, -16.8117599, -14.7829723, -1.0756764, 1.0710092
1: -16.3226395, -14.1662626, -16.2504559, -14.1760826, -1.4596510, 1.4465084
2: -12.1072292, -10.5934496, -12.0781193, -10.6079550, -1.0658634, 1.0612369
3: -11.9998159, -10.3451881, -11.9937553, -10.3724556, -1.3454847, 1.3544569
4: -2.2835135, -1.1341355, -2.2567663, -1.1375749, -0.9389687, 0.9261456
5: -8.1314659, -6.6055527, -8.1264286, -6.6248889, -0.8777075, 0.8820844
6: -16.8826714, -15.0701351, -16.8341293, -15.0735855, -0.9814804, 0.9704781
7: -6.8180022, -5.2145395, -6.8134255, -5.2176728, -1.2096519, 1.2079873
8: -3.6812420, -2.3537087, -3.6332235, -2.3546948, -1.2549515, 1.2408323
9: 5.4961758, 6.7154632, 5.4982548, 6.7104034, -0.9134240, 0.9165447

Time for backsubstitution: 24.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5759
type: A, layer: 1, pos: 5759
type: B, layer: 1, pos: 6236
type: A, layer: 1, pos: 6236
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 145
type: B, layer: 1, pos: 145
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 5759

### Candidate
type: A, layer: 1, pos: 5759

### Candidate
type: B, layer: 1, pos: 6236

## Relational analysis of NS_B1_A1_B1_A2_B1

### Relational analysis result of NS_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5246415, upper bound: 0.5241725
time: 6.09 seconds

## Relational analysis of NS_B1_A1_B1_A2_B2

### Relational analysis result of NS_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5248900, upper bound: 0.5248915
time: 4.79 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 58.08 + 542.01 = 600.10 seconds
