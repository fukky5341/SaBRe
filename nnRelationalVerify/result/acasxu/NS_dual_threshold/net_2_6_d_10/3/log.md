## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_6.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 3)
Time budget: 420 seconds
Split limit: 100
Threshold: 23.9931544845


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315)
1: (-1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685)
2: (-1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412)
3: (-1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150)
4: (-1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.27 + 1.29 = 3.55 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -24.1137231, upper bound: 24.1137231

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0791164, upper bound: 24.1137231
time: 0.40 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0792828, upper bound: 24.0792828
time: 0.40 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.00 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.00
Output dim: 0, lower bound: -24.0791164, upper bound: 24.1137231
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.00
Output dim: 0, lower bound: -24.0792828, upper bound: 24.0792828

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -5.1219692, 7.1375046, -9.5559692, 16.9026623, -22.0246258, 16.6934738
1: -0.8046948, 1.1233151, -1.9498287, 2.4651399, -3.2698345, 3.0731432
2: -0.6832222, 0.9289821, -1.5350443, 2.0507975, -2.7340198, 2.4640257
3: -0.6806831, 1.6734771, -1.4968777, 3.7269382, -4.4076204, 3.1703541
4: -0.6897084, 1.2417758, -1.5394063, 2.7020743, -3.3917828, 2.7811821

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0320068, upper bound: 24.0102485
time: 0.36 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0143030, upper bound: 24.0109569
time: 0.43 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -8.8194637, 15.1975842, -9.5559692, 16.9026623, -25.7221241, 24.7535534
1: -1.7194341, 2.2191176, -1.9498287, 2.4651399, -4.1845741, 4.1689463
2: -1.3808433, 1.8490930, -1.5350443, 2.0507975, -3.4316406, 3.3841364
3: -1.3433540, 3.3726079, -1.4968777, 3.7269382, -5.0702925, 4.8694839
4: -1.3641208, 2.4432526, -1.5394063, 2.7020743, -4.0661950, 3.9826589

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0383109, upper bound: 24.0198988
time: 0.42 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0206071, upper bound: 24.0206071
time: 0.43 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.17 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.17
Output dim: 0, lower bound: -24.0320068, upper bound: 24.0102485
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.17
Output dim: 0, lower bound: -24.0143030, upper bound: 24.0109569
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.17
Output dim: 0, lower bound: -24.0383109, upper bound: 24.0198988
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.17
Output dim: 0, lower bound: -24.0206071, upper bound: 24.0206071

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -5.1219692, 7.1375046, -4.4695539, 6.7397461, -11.8617153, 11.6070585
1: -0.8046948, 1.1233151, -0.7916995, 1.0438734, -1.8485682, 1.9150146
2: -0.6832222, 0.9289821, -0.6411211, 0.8345411, -1.5177631, 1.5701032
3: -0.6806831, 1.6734771, -0.6209559, 1.5210140, -2.2016971, 2.2944324
4: -0.6897084, 1.2417758, -0.6379681, 1.1251349, -1.8148433, 1.8797438

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_B1

### Relational analysis result of NS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0320068, upper bound: 24.0100951
time: 0.38 seconds

## Relational analysis of NS_A1_B1_B2

### Relational analysis result of NS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0320068, upper bound: 24.0102485
time: 0.38 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -5.1219692, 7.1375046, -8.0110874, 13.6429243, -18.7648869, 15.1485920
1: -0.8046948, 1.1233151, -1.5030907, 1.9807596, -2.7854543, 2.6264055
2: -0.6832222, 0.9289821, -1.2374809, 1.6453836, -2.3286054, 2.1664629
3: -0.6806831, 1.6734771, -1.1593255, 3.0259297, -3.7066128, 2.8328023
4: -0.6897084, 1.2417758, -1.1869328, 2.1908948, -2.8806033, 2.4287086

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_B1

### Relational analysis result of NS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0046527, upper bound: 24.0046527
time: 0.39 seconds

## Relational analysis of NS_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0046527, upper bound: 24.0109569
time: 0.37 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -8.8194637, 15.1975842, -4.4695539, 6.7397461, -15.5592079, 19.6671371
1: -1.7194341, 2.2191176, -0.7916995, 1.0438734, -2.7633076, 3.0108168
2: -1.3808433, 1.8490930, -0.6411211, 0.8345411, -2.2153842, 2.4902136
3: -1.3433540, 3.3726079, -0.6209559, 1.5210140, -2.8643677, 3.9935639
4: -1.3641208, 2.4432526, -0.6379681, 1.1251349, -2.4892557, 3.0812204

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0191958, upper bound: 24.0191958
time: 0.40 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0191958, upper bound: 24.0191958
time: 0.39 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -8.8194637, 15.1975842, -8.0110874, 13.6429243, -22.4623852, 23.2086716
1: -1.7194341, 2.2191176, -1.5030907, 1.9807596, -3.7001939, 3.7222083
2: -1.3808433, 1.8490930, -1.2374809, 1.6453836, -3.0262265, 3.0865736
3: -1.3433540, 3.3726079, -1.1593255, 3.0259297, -4.3692832, 4.5319328
4: -1.3641208, 2.4432526, -1.1869328, 2.1908948, -3.5550153, 3.6301854

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0191958, upper bound: 24.0206071
time: 0.41 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0191958, upper bound: 24.0206071
time: 0.42 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.16 seconds
NS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 3.16
Output dim: 0, lower bound: -24.0320068, upper bound: 24.0100951
NS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 3.16
Output dim: 0, lower bound: -24.0320068, upper bound: 24.0102485
NS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 3.16
Output dim: 0, lower bound: -24.0046527, upper bound: 24.0046527
NS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 3.16
Output dim: 0, lower bound: -24.0046527, upper bound: 24.0109569
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.16
Output dim: 0, lower bound: -24.0191958, upper bound: 24.0191958
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.16
Output dim: 0, lower bound: -24.0191958, upper bound: 24.0191958
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.16
Output dim: 0, lower bound: -24.0191958, upper bound: 24.0206071
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.16
Output dim: 0, lower bound: -24.0191958, upper bound: 24.0206071

## BFS NS instance: NS_A1_B1_B1

### Backsubstitution after applying NS history:
0: -5.1219692, 7.1375046, -2.5863075, 2.5088634, -7.6308312, 9.7238121
1: -0.8046948, 1.1233151, -0.3424560, 0.4800463, -1.2847412, 1.4657710
2: -0.6832222, 0.9289821, -0.2596823, 0.3596359, -1.0428581, 1.1886644
3: -0.6806831, 1.6734771, -0.2769262, 0.6608293, -1.3415124, 1.9504025
4: -0.6897084, 1.2417758, -0.2771527, 0.4922103, -1.1819186, 1.5189284

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_B1_A1

### Relational analysis result of NS_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0320068, upper bound: 24.0100951
time: 0.42 seconds

## Relational analysis of NS_A1_B1_B1_A2

### Relational analysis result of NS_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0320068, upper bound: 24.0100951
time: 0.37 seconds

## BFS NS instance: NS_A1_B1_B2

### Backsubstitution after applying NS history:
0: -5.1219692, 7.1375046, -4.0708694, 5.6211839, -10.7431517, 11.2083721
1: -0.8046948, 1.1233151, -0.6731225, 0.9151498, -1.7198446, 1.7964375
2: -0.6832222, 0.9289821, -0.5482491, 0.7177234, -1.4009455, 1.4772313
3: -0.6806831, 1.6734771, -0.5329784, 1.3033334, -1.9840165, 2.2064545
4: -0.6897084, 1.2417758, -0.5411942, 0.9680334, -1.6577418, 1.7829698

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_B2_A1

### Relational analysis result of NS_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0320068, upper bound: 24.0102485
time: 0.40 seconds

## Relational analysis of NS_A1_B1_B2_A2

### Relational analysis result of NS_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0320068, upper bound: 24.0102485
time: 0.38 seconds

## BFS NS instance: NS_A1_B2_B1

### Backsubstitution after applying NS history:
0: -5.1219692, 7.1375046, -4.1232567, 5.1132369, -10.2352066, 11.2607603
1: -0.8046948, 1.1233151, -0.5457485, 0.8533046, -1.6579993, 1.6690636
2: -0.6832222, 0.9289821, -0.4990431, 0.6817743, -1.3649962, 1.4280252
3: -0.6806831, 1.6734771, -0.4728078, 1.2374785, -1.9181616, 2.1462839
4: -0.6897084, 1.2417758, -0.4858359, 0.9272271, -1.6169355, 1.7276117

Time for backsubstitution: 2.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_B1_A1

### Relational analysis result of NS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0046527, upper bound: 24.0046527
time: 0.39 seconds

## Relational analysis of NS_A1_B2_B1_A2

### Relational analysis result of NS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0046527, upper bound: 24.0046527
time: 0.38 seconds

## BFS NS instance: NS_A1_B2_B2

### Backsubstitution after applying NS history:
0: -5.1219692, 7.1375046, -7.5259757, 12.4696693, -17.5916328, 14.6634779
1: -0.8046948, 1.1233151, -1.3758048, 1.8236271, -2.6283219, 2.4991193
2: -0.6832222, 0.9289821, -1.1365285, 1.5210134, -2.2042356, 2.0655107
3: -0.6806831, 1.6734771, -1.0658861, 2.7853112, -3.4659944, 2.7393627
4: -0.6897084, 1.2417758, -1.0868741, 2.0250101, -2.7147183, 2.3286500

Time for backsubstitution: 2.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_B2_A1

### Relational analysis result of NS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0046527, upper bound: 24.0109569
time: 0.38 seconds

## Relational analysis of NS_A1_B2_B2_A2

### Relational analysis result of NS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0046527, upper bound: 24.0095456
time: 0.38 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -4.0708694, 5.6211839, -4.4695539, 6.7397461, -10.8106155, 10.0907373
1: -0.6731225, 0.9151498, -0.7916995, 1.0438734, -1.7169960, 1.7068492
2: -0.5482491, 0.7177234, -0.6411211, 0.8345411, -1.3827901, 1.3588445
3: -0.5329784, 1.3033334, -0.6209559, 1.5210140, -2.0539923, 1.9242892
4: -0.5411942, 0.9680334, -0.6379681, 1.1251349, -1.6663291, 1.6060016

Time for backsubstitution: 2.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0320068, upper bound: 24.0197454
time: 0.42 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0368996, upper bound: 24.0198988
time: 0.42 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -7.5259757, 12.4696693, -4.4695539, 6.7397461, -14.2657223, 16.9392242
1: -1.3758048, 1.8236271, -0.7916995, 1.0438734, -2.4196782, 2.6153266
2: -1.1365285, 1.5210134, -0.6411211, 0.8345411, -1.9710696, 2.1621344
3: -1.0658861, 2.7853112, -0.6209559, 1.5210140, -2.5869002, 3.4062672
4: -1.0868741, 2.0250101, -0.6379681, 1.1251349, -2.2120090, 2.6629777

Time for backsubstitution: 2.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0320068, upper bound: 24.0197454
time: 0.43 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0368996, upper bound: 24.0198988
time: 0.41 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -4.0708694, 5.6211839, -8.0110874, 13.6429243, -17.7137890, 13.6322708
1: -0.6731225, 0.9151498, -1.5030907, 1.9807596, -2.6538820, 2.4182405
2: -0.5482491, 0.7177234, -1.2374809, 1.6453836, -2.1936321, 1.9552042
3: -0.5329784, 1.3033334, -1.1593255, 3.0259297, -3.5589077, 2.4626589
4: -0.5411942, 0.9680334, -1.1869328, 2.1908948, -2.7320890, 2.1549659

Time for backsubstitution: 2.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0095456, upper bound: 24.0143030
time: 0.43 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0095456, upper bound: 24.0194971
time: 0.48 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -7.5259757, 12.4696693, -8.0110874, 13.6429243, -21.1688976, 20.4807549
1: -1.3758048, 1.8236271, -1.5030907, 1.9807596, -3.3565645, 3.3267176
2: -1.1365285, 1.5210134, -1.2374809, 1.6453836, -2.7819118, 2.7584944
3: -1.0658861, 2.7853112, -1.1593255, 3.0259297, -4.0918159, 3.9446368
4: -1.0868741, 2.0250101, -1.1869328, 2.1908948, -3.2777691, 3.2119429

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0095456, upper bound: 24.0143030
time: 0.43 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0095456, upper bound: 24.0191958
time: 0.42 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.72 seconds
NS_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.72
Output dim: 0, lower bound: -24.0320068, upper bound: 24.0100951
NS_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.72
Output dim: 0, lower bound: -24.0320068, upper bound: 24.0100951
NS_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.72
Output dim: 0, lower bound: -24.0320068, upper bound: 24.0102485
NS_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.72
Output dim: 0, lower bound: -24.0320068, upper bound: 24.0102485
NS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.72
Output dim: 0, lower bound: -24.0046527, upper bound: 24.0046527
NS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.72
Output dim: 0, lower bound: -24.0046527, upper bound: 24.0046527
NS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.72
Output dim: 0, lower bound: -24.0046527, upper bound: 24.0109569
NS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.72
Output dim: 0, lower bound: -24.0046527, upper bound: 24.0095456
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.72
Output dim: 0, lower bound: -24.0320068, upper bound: 24.0197454
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.72
Output dim: 0, lower bound: -24.0368996, upper bound: 24.0198988
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.72
Output dim: 0, lower bound: -24.0320068, upper bound: 24.0197454
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.72
Output dim: 0, lower bound: -24.0368996, upper bound: 24.0198988
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.72
Output dim: 0, lower bound: -24.0095456, upper bound: 24.0143030
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.72
Output dim: 0, lower bound: -24.0095456, upper bound: 24.0194971
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.72
Output dim: 0, lower bound: -24.0095456, upper bound: 24.0143030
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.72
Output dim: 0, lower bound: -24.0095456, upper bound: 24.0191958

## BFS NS instance: NS_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -2.5863075, 2.5088634, -2.5863075, 2.5088634, -5.0951705, 5.0951705
1: -0.3424560, 0.4800463, -0.3424560, 0.4800463, -0.8225023, 0.8225023
2: -0.2596823, 0.3596359, -0.2596823, 0.3596359, -0.6193182, 0.6193182
3: -0.2769262, 0.6608293, -0.2769262, 0.6608293, -0.9377555, 0.9377555
4: -0.2771527, 0.4922103, -0.2771527, 0.4922103, -0.7693629, 0.7693629

Time for backsubstitution: 2.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## BFS NS instance: NS_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -4.1232567, 5.1132369, -2.5863075, 2.5088634, -6.6321197, 7.6995444
1: -0.5457485, 0.8533046, -0.3424560, 0.4800463, -1.0257949, 1.1957605
2: -0.4990431, 0.6817743, -0.2596823, 0.3596359, -0.8586790, 0.9414566
3: -0.4728078, 1.2374785, -0.2769262, 0.6608293, -1.1336370, 1.5144048
4: -0.4858359, 0.9272271, -0.2771527, 0.4922103, -0.9780462, 1.2043798

Time for backsubstitution: 2.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_B1_A2_B1

### Relational analysis result of NS_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0828110, upper bound: 23.9877249
time: 0.40 seconds

## Relational analysis of NS_A1_B1_B1_A2_B2

### Relational analysis result of NS_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0924361, upper bound: 24.0100765
time: 0.44 seconds

## BFS NS instance: NS_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -2.5863075, 2.5088634, -4.0708694, 5.6211839, -8.2074909, 6.5797319
1: -0.3424560, 0.4800463, -0.6731225, 0.9151498, -1.2576058, 1.1531688
2: -0.2596823, 0.3596359, -0.5482491, 0.7177234, -0.9774057, 0.9078849
3: -0.2769262, 0.6608293, -0.5329784, 1.3033334, -1.5802596, 1.1938077
4: -0.2771527, 0.4922103, -0.5411942, 0.9680334, -1.2451861, 1.0334044

Time for backsubstitution: 2.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_B2_A1_B1

### Relational analysis result of NS_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0320068, upper bound: 24.0102485
time: 0.38 seconds

## Relational analysis of NS_A1_B1_B2_A1_B2

### Relational analysis result of NS_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0294753, upper bound: 24.0096615
time: 0.42 seconds

## BFS NS instance: NS_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -4.1232567, 5.1132369, -4.0708694, 5.6211839, -9.7444391, 9.1841068
1: -0.5457485, 0.8533046, -0.6731225, 0.9151498, -1.4608984, 1.5264270
2: -0.4990431, 0.6817743, -0.5482491, 0.7177234, -1.2167665, 1.2300234
3: -0.4728078, 1.2374785, -0.5329784, 1.3033334, -1.7761412, 1.7704569
4: -0.4858359, 0.9272271, -0.5411942, 0.9680334, -1.4538693, 1.4684212

Time for backsubstitution: 2.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_B2_A2_B1

### Relational analysis result of NS_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0320068, upper bound: 24.0102485
time: 0.38 seconds

## Relational analysis of NS_A1_B1_B2_A2_B2

### Relational analysis result of NS_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0294753, upper bound: 24.0096615
time: 0.40 seconds

## BFS NS instance: NS_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -2.5863075, 2.5088634, -4.1232567, 5.1132369, -7.6995444, 6.6321197
1: -0.3424560, 0.4800463, -0.5457485, 0.8533046, -1.1957604, 1.0257949
2: -0.2596823, 0.3596359, -0.4990431, 0.6817743, -0.9414564, 0.8586790
3: -0.2769262, 0.6608293, -0.4728078, 1.2374785, -1.5144048, 1.1336371
4: -0.2771527, 0.4922103, -0.4858359, 0.9272271, -1.2043798, 0.9780462

Time for backsubstitution: 2.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 41

## BFS NS instance: NS_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -4.1232567, 5.1132369, -4.1232567, 5.1132369, -9.2364941, 9.2364941
1: -0.5457485, 0.8533046, -0.5457485, 0.8533046, -1.3990531, 1.3990531
2: -0.4990431, 0.6817743, -0.4990431, 0.6817743, -1.1808174, 1.1808174
3: -0.4728078, 1.2374785, -0.4728078, 1.2374785, -1.7102863, 1.7102863
4: -0.4858359, 0.9272271, -0.4858359, 0.9272271, -1.4130629, 1.4130629

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## BFS NS instance: NS_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -2.5863075, 2.5088634, -7.5259757, 12.4696693, -15.0559731, 10.0348396
1: -0.3424560, 0.4800463, -1.3758048, 1.8236271, -2.1660829, 1.8558511
2: -0.2596823, 0.3596359, -1.1365285, 1.5210134, -1.7806956, 1.4961644
3: -0.2769262, 0.6608293, -1.0658861, 2.7853112, -3.0622373, 1.7267153
4: -0.2771527, 0.4922103, -1.0868741, 2.0250101, -2.3021629, 1.5790844

Time for backsubstitution: 2.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

## BFS NS instance: NS_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -4.1232567, 5.1132369, -7.5259757, 12.4696693, -16.5929241, 12.6392126
1: -0.5457485, 0.8533046, -1.3758048, 1.8236271, -2.3693757, 2.2291093
2: -0.4990431, 0.6817743, -1.1365285, 1.5210134, -2.0200565, 1.8183026
3: -0.4728078, 1.2374785, -1.0658861, 2.7853112, -3.2581189, 2.3033648
4: -0.4858359, 0.9272271, -1.0868741, 2.0250101, -2.5108459, 2.0141013

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 41

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -4.0708694, 5.6211839, -2.5863075, 2.5088634, -6.5797319, 8.2074909
1: -0.6731225, 0.9151498, -0.3424560, 0.4800463, -1.1531688, 1.2576057
2: -0.5482491, 0.7177234, -0.2596823, 0.3596359, -0.9078850, 0.9774057
3: -0.5329784, 1.3033334, -0.2769262, 0.6608293, -1.1938077, 1.5802596
4: -0.5411942, 0.9680334, -0.2771527, 0.4922103, -1.0334045, 1.2451861

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0350711, upper bound: 24.0368797
time: 0.44 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0344841, upper bound: 24.0343483
time: 0.47 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -4.0708694, 5.6211839, -4.0708694, 5.6211839, -9.6920519, 9.6920519
1: -0.6731225, 0.9151498, -0.6731225, 0.9151498, -1.5882723, 1.5882723
2: -0.5482491, 0.7177234, -0.5482491, 0.7177234, -1.2659724, 1.2659724
3: -0.5329784, 1.3033334, -0.5329784, 1.3033334, -1.8363118, 1.8363117
4: -0.5411942, 0.9680334, -0.5411942, 0.9680334, -1.5092275, 1.5092275

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0350711, upper bound: 24.0368797
time: 0.43 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0344841, upper bound: 24.0343483
time: 0.41 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -7.5259757, 12.4696693, -2.5863075, 2.5088634, -10.0348396, 15.0559731
1: -1.3758048, 1.8236271, -0.3424560, 0.4800463, -1.8558511, 2.1660829
2: -1.1365285, 1.5210134, -0.2596823, 0.3596359, -1.4961644, 1.7806957
3: -1.0658861, 2.7853112, -0.2769262, 0.6608293, -1.7267154, 3.0622373
4: -1.0868741, 2.0250101, -0.2771527, 0.4922103, -1.5790844, 2.3021629

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0357795, upper bound: 24.0191759
time: 0.46 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0357795, upper bound: 24.0185173
time: 0.41 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -7.5259757, 12.4696693, -4.0708694, 5.6211839, -13.1471586, 16.5405350
1: -1.3758048, 1.8236271, -0.6731225, 0.9151498, -2.2909546, 2.4967496
2: -1.1365285, 1.5210134, -0.5482491, 0.7177234, -1.8542519, 2.0692623
3: -1.0658861, 2.7853112, -0.5329784, 1.3033334, -2.3692195, 3.3182893
4: -1.0868741, 2.0250101, -0.5411942, 0.9680334, -2.0549076, 2.5662041

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 41

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -4.0708694, 5.6211839, -4.1232567, 5.1132369, -9.1841068, 9.7444391
1: -0.6731225, 0.9151498, -0.5457485, 0.8533046, -1.5264270, 1.4608984
2: -0.5482491, 0.7177234, -0.4990431, 0.6817743, -1.2300234, 1.2167665
3: -0.5329784, 1.3033334, -0.4728078, 1.2374785, -1.7704567, 1.7761412
4: -0.5411942, 0.9680334, -0.4858359, 0.9272271, -1.4684212, 1.4538693

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0102485, upper bound: 24.0320068
time: 0.39 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0096615, upper bound: 24.0294753
time: 0.40 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -4.0708694, 5.6211839, -7.5259757, 12.4696693, -16.5405369, 13.1471596
1: -0.6731225, 0.9151498, -1.3758048, 1.8236271, -2.4967496, 2.2909546
2: -0.5482491, 0.7177234, -1.1365285, 1.5210134, -2.0692623, 1.8542519
3: -0.5329784, 1.3033334, -1.0658861, 2.7853112, -3.3182893, 2.3692195
4: -0.5411942, 0.9680334, -1.0868741, 2.0250101, -2.5662041, 2.0549076

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 41

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -7.5259757, 12.4696693, -4.1232567, 5.1132369, -12.6392126, 16.5929241
1: -1.3758048, 1.8236271, -0.5457485, 0.8533046, -2.2291093, 2.3693757
2: -1.1365285, 1.5210134, -0.4990431, 0.6817743, -1.8183026, 2.0200565
3: -1.0658861, 2.7853112, -0.4728078, 1.2374785, -2.3033645, 3.2581186
4: -1.0868741, 2.0250101, -0.4858359, 0.9272271, -2.0141013, 2.5108459

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0109569, upper bound: 24.0143030
time: 0.48 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0109569, upper bound: 24.0136444
time: 0.44 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -7.5259757, 12.4696693, -7.5259757, 12.4696693, -19.9956417, 19.9956417
1: -1.3758048, 1.8236271, -1.3758048, 1.8236271, -3.1994319, 3.1994319
2: -1.1365285, 1.5210134, -1.1365285, 1.5210134, -2.6575418, 2.6575418
3: -1.0658861, 2.7853112, -1.0658861, 2.7853112, -3.8511972, 3.8511972
4: -1.0868741, 2.0250101, -1.0868741, 2.0250101, -3.1118841, 3.1118841

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.55 + 102.45 = 106.00 seconds
