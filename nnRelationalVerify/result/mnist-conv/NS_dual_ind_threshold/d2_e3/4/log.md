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
execution time: IAR + RelationalAnalysis = 22.75 + 34.97 = 57.72 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.5250911, upper bound: 0.5250914

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 5759
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 6236
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 145

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 845

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5247634, upper bound: 0.5248753
time: 4.48 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5247634, upper bound: 0.5247636
time: 4.66 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 9.35 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 9.35
Output dim: 9, lower bound: -0.5247634, upper bound: 0.5248753
NS_A2, status: Status.UNKNOWN, split count: 1, time: 9.35
Output dim: 9, lower bound: -0.5247634, upper bound: 0.5247636

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -16.8118744, -14.7827873, -16.8119717, -14.7808495, -1.0741301, 1.0719867
1: -16.2506256, -14.1728897, -16.2532425, -14.1725426, -1.4433937, 1.4461150
2: -12.0781336, -10.6066895, -12.0797310, -10.6063452, -1.0544605, 1.0551620
3: -11.9944878, -10.3722916, -11.9946671, -10.3721228, -1.3400803, 1.3400254
4: -2.2570481, -1.1362169, -2.2573123, -1.1360272, -0.9256430, 0.9246087
5: -8.1274128, -6.6248302, -8.1274948, -6.6219063, -0.8783860, 0.8752856
6: -16.8342171, -15.0713892, -16.8348351, -15.0712833, -0.9699345, 0.9708509
7: -6.8136673, -5.2175102, -6.8139114, -5.2144313, -1.2103243, 1.2073922
8: -3.6335135, -2.3537788, -3.6339922, -2.3533592, -1.2437840, 1.2435365
9: 5.4980354, 6.7104521, 5.4977684, 6.7104955, -0.9121385, 0.9129844

Time for backsubstitution: 21.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 5759
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 6236
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 145

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 845

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5246738, upper bound: 0.5246752
time: 3.58 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5246738, upper bound: 0.5246752
time: 3.48 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -16.8185768, -14.7792740, -16.8120461, -14.7792988, -1.0826459, 1.0743632
1: -16.2566032, -14.1614799, -16.2553291, -14.1722746, -1.4470320, 1.4599409
2: -12.0813856, -10.5969620, -12.0809984, -10.6060772, -1.0573564, 1.0673470
3: -11.9946308, -10.3698521, -11.9947958, -10.3720665, -1.3458939, 1.3434696
4: -2.2577202, -1.1360559, -2.2572832, -1.1358852, -0.9317465, 0.9228177
5: -8.1365108, -6.6192932, -8.1275625, -6.6195807, -0.8887172, 0.8776810
6: -16.8356094, -15.0689297, -16.8353291, -15.0712013, -0.9704189, 0.9740710
7: -6.8237934, -5.2111654, -6.8141031, -5.2119765, -1.2228208, 1.2118325
8: -3.6397457, -2.3528509, -3.6343646, -2.3530240, -1.2516899, 1.2462249
9: 5.4964356, 6.7102175, 5.4975581, 6.7103491, -0.9121141, 0.9177604

Time for backsubstitution: 20.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 5759
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 6236
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 145

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 845

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5246738, upper bound: 0.5247648
time: 3.98 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5246738, upper bound: 0.5247634
time: 3.54 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 27.89 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 27.89
Output dim: 9, lower bound: -0.5246738, upper bound: 0.5246752
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 27.89
Output dim: 9, lower bound: -0.5246738, upper bound: 0.5246752
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 27.89
Output dim: 9, lower bound: -0.5246738, upper bound: 0.5247648
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 27.89
Output dim: 9, lower bound: -0.5246738, upper bound: 0.5247634

## BFS NS instance: NS_A1_B1

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

Time for backsubstitution: 20.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5759
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 6236
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 145

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5759

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 524

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5246252, upper bound: 0.5248755
time: 3.51 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5246730, upper bound: 0.5248755
time: 3.67 seconds

## BFS NS instance: NS_A1_B2

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

Time for backsubstitution: 20.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5759
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 6236
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 145

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5759

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 524

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5246252, upper bound: 0.5248755
time: 3.46 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5246730, upper bound: 0.5248755
time: 3.48 seconds

## BFS NS instance: NS_A2_B1

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

Time for backsubstitution: 21.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5759
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 6236
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 145

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5759

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 524

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5246252, upper bound: 0.5247637
time: 3.64 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5246730, upper bound: 0.5247633
time: 4.43 seconds

## BFS NS instance: NS_A2_B2

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

Time for backsubstitution: 21.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5759
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 6236
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 145

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5759

## Relational analysis of NS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 524

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5246252, upper bound: 0.5247639
time: 3.64 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5246730, upper bound: 0.5247639
time: 3.61 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 34.66 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 34.66
Output dim: 9, lower bound: -0.5246252, upper bound: 0.5248755
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 34.66
Output dim: 9, lower bound: -0.5246730, upper bound: 0.5248755
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 34.66
Output dim: 9, lower bound: -0.5246252, upper bound: 0.5248755
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 34.66
Output dim: 9, lower bound: -0.5246730, upper bound: 0.5248755
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 34.66
Output dim: 9, lower bound: -0.5246252, upper bound: 0.5247637
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 34.66
Output dim: 9, lower bound: -0.5246730, upper bound: 0.5247633
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 34.66
Output dim: 9, lower bound: -0.5246252, upper bound: 0.5247639
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 34.66
Output dim: 9, lower bound: -0.5246730, upper bound: 0.5247639

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -16.8117599, -14.7829723, -16.8118744, -14.7827873, -1.0711722, 1.0711384
1: -16.2504559, -14.1760826, -16.2506256, -14.1728897, -1.4429235, 1.4398890
2: -12.0781193, -10.6079550, -12.0781336, -10.6066895, -1.0541425, 1.0527449
3: -11.9937553, -10.3724556, -11.9944878, -10.3722916, -1.3389530, 1.3395519
4: -2.2567663, -1.1375749, -2.2570481, -1.1362169, -0.9243474, 0.9231544
5: -8.1264286, -6.6248889, -8.1274128, -6.6248302, -0.8741722, 0.8751431
6: -16.8341293, -15.0735855, -16.8342171, -15.0713892, -0.9694791, 0.9675162
7: -6.8134255, -5.2176728, -6.8136673, -5.2175102, -1.2062745, 1.2062778
8: -3.6332235, -2.3546948, -3.6335135, -2.3537788, -1.2410040, 1.2400913
9: 5.4982548, 6.7104034, 5.4980354, 6.7104521, -0.9117608, 0.9119825

Time for backsubstitution: 20.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5759
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 6236
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 145

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 5759

### Candidate
type: B, layer: 1, pos: 524

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5248900, upper bound: 0.5248898
time: 7.22 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5248900, upper bound: 0.5249390
time: 6.73 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -16.8188763, -14.7826519, -16.8118763, -14.7827902, -1.0764256, 1.0845811
1: -16.3226395, -14.1662626, -16.2506275, -14.1729040, -1.4628830, 1.4523678
2: -12.1072292, -10.5934496, -12.0781336, -10.6066971, -1.0672908, 1.0616398
3: -11.9998159, -10.3451881, -11.9944849, -10.3722935, -1.3462849, 1.3552895
4: -2.2835135, -1.1341355, -2.2570477, -1.1362216, -0.9403346, 0.9273176
5: -8.1314659, -6.6055527, -8.1274099, -6.6248312, -0.8785529, 0.8831089
6: -16.8826714, -15.0701351, -16.8342190, -15.0713978, -0.9837971, 0.9785311
7: -6.8180022, -5.2145395, -6.8136659, -5.2175117, -1.2305574, 1.2089906
8: -3.6812420, -2.3537087, -3.6335130, -2.3537817, -1.2579327, 1.2620807
9: 5.4961758, 6.7154632, 5.4980359, 6.7104526, -0.9158750, 0.9168630

Time for backsubstitution: 21.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5759
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 6236
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 145

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5759

### Candidate
type: B, layer: 1, pos: 524

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5249389, upper bound: 0.5248901
time: 6.66 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5249389, upper bound: 0.5249389
time: 6.09 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -16.8117599, -14.7829723, -16.8185768, -14.7792740, -1.0746408, 1.0778670
1: -16.2504559, -14.1760826, -16.2566032, -14.1614799, -1.4543591, 1.4459872
2: -12.0781193, -10.6079550, -12.0813856, -10.5969620, -1.0651426, 1.0558386
3: -11.9937553, -10.3724556, -11.9946308, -10.3698521, -1.3429718, 1.3436518
4: -2.2567663, -1.1375749, -2.2577202, -1.1360559, -0.9244003, 0.9235749
5: -8.1264286, -6.6248889, -8.1365108, -6.6192932, -0.8797665, 0.8829286
6: -16.8341293, -15.0735855, -16.8356094, -15.0689297, -0.9718995, 0.9689798
7: -6.8134255, -5.2176728, -6.8237934, -5.2111654, -1.2127333, 1.2163458
8: -3.6332235, -2.3546948, -3.6397457, -2.3528509, -1.2425003, 1.2474380
9: 5.4982548, 6.7104034, 5.4964356, 6.7102175, -0.9117579, 0.9135361

Time for backsubstitution: 21.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5759
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 6236
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 145

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5759

### Candidate
type: B, layer: 1, pos: 524

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5247143, upper bound: 0.5248273
time: 4.28 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5247143, upper bound: 0.5248755
time: 3.65 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -16.8188763, -14.7826519, -16.8185730, -14.7792721, -1.0798945, 1.0854938
1: -16.3226395, -14.1662626, -16.2566013, -14.1614914, -1.4655352, 1.4584670
2: -12.1072292, -10.5934496, -12.0813847, -10.5969667, -1.0713773, 1.0647330
3: -11.9998159, -10.3451881, -11.9946289, -10.3698549, -1.3503056, 1.3574414
4: -2.2835135, -1.1341355, -2.2577209, -1.1360621, -0.9411364, 0.9277382
5: -8.1314659, -6.6055527, -8.1365089, -6.6192956, -0.8841467, 0.8836205
6: -16.8826714, -15.0701351, -16.8356113, -15.0689383, -0.9843953, 0.9799957
7: -6.8180022, -5.2145395, -6.8237944, -5.2111692, -1.2361841, 1.2190609
8: -3.6812420, -2.3537087, -3.6397452, -2.3528552, -1.2594552, 1.2659688
9: 5.4961758, 6.7154632, 5.4964361, 6.7102151, -0.9158731, 0.9184184

Time for backsubstitution: 21.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5759
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 6236
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 145

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5759

### Candidate
type: B, layer: 1, pos: 524

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5247624, upper bound: 0.5248262
time: 4.69 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5247624, upper bound: 0.5248744
time: 4.17 seconds

## BFS NS instance: NS_A2_B1_A1

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

Time for backsubstitution: 21.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5759
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 6236
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 145

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 5759

### Candidate
type: B, layer: 1, pos: 524

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5248259, upper bound: 0.5247157
time: 3.67 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5248259, upper bound: 0.5247637
time: 3.68 seconds

## BFS NS instance: NS_A2_B1_A2

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

Time for backsubstitution: 21.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5759
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 6236
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 145

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5759

### Candidate
type: B, layer: 1, pos: 524

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5248742, upper bound: 0.5247157
time: 3.73 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5248742, upper bound: 0.5247637
time: 3.84 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -16.8184566, -14.7794638, -16.8185768, -14.7792740, -1.0741520, 1.0741215
1: -16.2564297, -14.1646700, -16.2566032, -14.1614799, -1.4501371, 1.4471059
2: -12.0813694, -10.5982246, -12.0813856, -10.5969620, -1.0645971, 1.0632043
3: -11.9938993, -10.3700199, -11.9946308, -10.3698521, -1.3474751, 1.3480716
4: -2.2574382, -1.1374135, -2.2577202, -1.1360559, -0.9319530, 0.9307599
5: -8.1355286, -6.6193523, -8.1365108, -6.6192932, -0.8768559, 0.8778250
6: -16.8355217, -15.0711288, -16.8356094, -15.0689297, -0.9702482, 0.9682777
7: -6.8235569, -5.2113314, -6.8237934, -5.2111654, -1.2147331, 1.2147346
8: -3.6394529, -2.3537650, -3.6397457, -2.3528509, -1.2492857, 1.2483697
9: 5.4966555, 6.7101684, 5.4964356, 6.7102175, -0.9183154, 0.9185367

Time for backsubstitution: 21.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5759
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 6236
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 145

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5759

### Candidate
type: B, layer: 1, pos: 524

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5246249, upper bound: 0.5247159
time: 3.89 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5246249, upper bound: 0.5247639
time: 3.72 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -16.8255730, -14.7791367, -16.8185730, -14.7792721, -1.0794086, 1.0876510
1: -16.3286018, -14.1548710, -16.2566013, -14.1614914, -1.4695544, 1.4595766
2: -12.1104774, -10.5837841, -12.0813847, -10.5969667, -1.0741568, 1.0720539
3: -12.0000219, -10.3426847, -11.9946289, -10.3698549, -1.3547592, 1.3618040
4: -2.2841709, -1.1339774, -2.2577209, -1.1360621, -0.9446564, 0.9349241
5: -8.1405516, -6.6000161, -8.1365089, -6.6192956, -0.8812332, 0.8868616
6: -16.8840809, -15.0676727, -16.8356113, -15.0689383, -0.9861867, 0.9792910
7: -6.8281164, -5.2081923, -6.8237944, -5.2111692, -1.2390089, 1.2174549
8: -3.6875019, -2.3527803, -3.6397452, -2.3528552, -1.2642369, 1.2678714
9: 5.4945602, 6.7152224, 5.4964361, 6.7102151, -0.9224329, 0.9234149

Time for backsubstitution: 21.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5759
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 6236
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 145

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 5759

### Candidate
type: B, layer: 1, pos: 524

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5246730, upper bound: 0.5247158
time: 3.72 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5246730, upper bound: 0.5247639
time: 3.86 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 29.39 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 29.39
Output dim: 9, lower bound: -0.5248900, upper bound: 0.5248898
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 29.39
Output dim: 9, lower bound: -0.5248900, upper bound: 0.5249390
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 29.39
Output dim: 9, lower bound: -0.5249389, upper bound: 0.5248901
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 29.39
Output dim: 9, lower bound: -0.5249389, upper bound: 0.5249389
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 29.39
Output dim: 9, lower bound: -0.5247143, upper bound: 0.5248273
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 29.39
Output dim: 9, lower bound: -0.5247143, upper bound: 0.5248755
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 29.39
Output dim: 9, lower bound: -0.5247624, upper bound: 0.5248262
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 29.39
Output dim: 9, lower bound: -0.5247624, upper bound: 0.5248744
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 29.39
Output dim: 9, lower bound: -0.5248259, upper bound: 0.5247157
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 29.39
Output dim: 9, lower bound: -0.5248259, upper bound: 0.5247637
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 29.39
Output dim: 9, lower bound: -0.5248742, upper bound: 0.5247157
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 29.39
Output dim: 9, lower bound: -0.5248742, upper bound: 0.5247637
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 29.39
Output dim: 9, lower bound: -0.5246249, upper bound: 0.5247159
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 29.39
Output dim: 9, lower bound: -0.5246249, upper bound: 0.5247639
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 29.39
Output dim: 9, lower bound: -0.5246730, upper bound: 0.5247158
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 29.39
Output dim: 9, lower bound: -0.5246730, upper bound: 0.5247639

## BFS NS instance: NS_A1_B1_A1_B1

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

Time for backsubstitution: 21.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5759
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 6236
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 145

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 5759

### Candidate
type: A, layer: 1, pos: 550

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5248902, upper bound: 0.5248826
time: 4.67 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5248902, upper bound: 0.5248913
time: 4.44 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -16.8117599, -14.7829723, -16.8188763, -14.7826519, -1.0710092, 1.0756760
1: -16.2504559, -14.1760826, -16.3226395, -14.1662626, -1.4465079, 1.4596510
2: -12.0781193, -10.6079550, -12.1072292, -10.5934496, -1.0612369, 1.0658634
3: -11.9937553, -10.3724556, -11.9998159, -10.3451881, -1.3544569, 1.3454847
4: -2.2567663, -1.1375749, -2.2835135, -1.1341355, -0.9261456, 0.9389689
5: -8.1264286, -6.6248889, -8.1314659, -6.6055527, -0.8820844, 0.8777075
6: -16.8341293, -15.0735855, -16.8826714, -15.0701351, -0.9704781, 0.9814801
7: -6.8134255, -5.2176728, -6.8180022, -5.2145395, -1.2079878, 1.2096519
8: -3.6332235, -2.3546948, -3.6812420, -2.3537087, -1.2408323, 1.2549510
9: 5.4982548, 6.7104034, 5.4961758, 6.7154632, -0.9165444, 0.9134240

Time for backsubstitution: 21.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5759
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 6236
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 145

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 5759

### Candidate
type: A, layer: 1, pos: 550

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5248902, upper bound: 0.5249316
time: 4.31 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5248902, upper bound: 0.5249402
time: 4.81 seconds

## BFS NS instance: NS_A1_B1_A2_B1

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

Time for backsubstitution: 22.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5759
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 6236
type: A, layer: 1, pos: 145
type: A, layer: 1, pos: 119

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 5759

### Candidate
type: A, layer: 1, pos: 550

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5249389, upper bound: 0.5248811
time: 5.35 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5249389, upper bound: 0.5248901
time: 5.58 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 57.72 + 543.41 = 601.13 seconds
