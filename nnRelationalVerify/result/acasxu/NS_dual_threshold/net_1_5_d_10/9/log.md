## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_5.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 9)
Time budget: 420 seconds
Split limit: 100
Threshold: 27.7691976323


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604)
1: (-11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856)
2: (-9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369)
3: (-10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898)
4: (-8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.74 + 1.60 = 2.34 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -27.8527559, upper bound: 27.8527559

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8398031, upper bound: 27.8414878
time: 0.45 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8373553, upper bound: 27.8374351
time: 0.46 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 0.99 seconds
NS_B1, status: Status.UNKNOWN, split count: 1, time: 0.99
Output dim: 0, lower bound: -27.8398031, upper bound: 27.8414878
NS_B2, status: Status.UNKNOWN, split count: 1, time: 0.99
Output dim: 0, lower bound: -27.8373553, upper bound: 27.8374351

## BFS NS instance: NS_B1

### Backsubstitution after applying NS history:
0: -6.2470675, 21.3258667, -5.4061508, 18.7729778, -25.0200405, 26.7320137
1: -10.2634687, 21.7926369, -8.9430733, 19.1648998, -29.4283676, 30.7357101
2: -8.4031277, 23.5036335, -7.2751994, 20.7168579, -29.1199856, 30.7788334
3: -9.1219254, 32.2972565, -7.9728169, 28.5126152, -37.6345406, 40.2700729
4: -7.4349117, 30.5508499, -6.4483242, 26.9256763, -34.3605881, 36.9991760

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B1_B1

### Relational analysis result of NS_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7954689, upper bound: 27.8041228
time: 0.50 seconds

## Relational analysis of NS_B1_B2

### Relational analysis result of NS_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -26.7744889, upper bound: 26.7768476
time: 0.71 seconds

## BFS NS instance: NS_B2

### Backsubstitution after applying NS history:
0: -6.9256406, 23.3276978, -6.6130686, 22.3832588, -29.3088989, 29.9407654
1: -11.3224821, 23.8662930, -10.8148479, 22.8833485, -34.2058220, 34.6811409
2: -9.3035965, 25.6878433, -8.8901510, 24.6683903, -33.9719849, 34.5779953
3: -10.0468111, 35.2414093, -9.6036949, 33.8211365, -43.8679466, 44.8451042
4: -8.2302294, 33.4003677, -7.8671532, 32.0281372, -40.2583618, 41.2675209

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B2_A1

### Relational analysis result of NS_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8374351, upper bound: 27.8373553
time: 0.87 seconds

## Relational analysis of NS_B2_A2

### Relational analysis result of NS_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8374351, upper bound: 27.8374351
time: 0.70 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.29 seconds
NS_B1_B1, status: Status.UNKNOWN, split count: 2, time: 2.29
Output dim: 0, lower bound: -27.7954689, upper bound: 27.8041228
NS_B1_B2, status: Status.VERIFIED, split count: 2, time: 2.29
Output dim: 0, lower bound: -26.7744889, upper bound: 26.7768476
NS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 2.29
Output dim: 0, lower bound: -27.8374351, upper bound: 27.8373553
NS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 2.29
Output dim: 0, lower bound: -27.8374351, upper bound: 27.8374351

## BFS NS instance: NS_B1_B1

### Backsubstitution after applying NS history:
0: -6.0153055, 20.6205425, -4.9898529, 17.4711838, -23.4864883, 25.6103954
1: -9.8962240, 21.0645370, -8.2712097, 17.8242245, -27.7204456, 29.3357468
2: -8.0930872, 22.7344818, -6.7083406, 19.2912483, -27.3843346, 29.4428215
3: -8.7982836, 31.2478485, -7.3757353, 26.5473442, -35.3456268, 38.6235847
4: -7.1616430, 29.5352783, -5.9454603, 25.0227203, -32.1843643, 35.4807396

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B1_B1_A1

### Relational analysis result of NS_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7886400, upper bound: 27.7928469
time: 1.09 seconds

## Relational analysis of NS_B1_B1_A2

### Relational analysis result of NS_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7931172, upper bound: 27.7975445
time: 0.75 seconds

## BFS NS instance: NS_B2_A1

### Backsubstitution after applying NS history:
0: -5.4061508, 18.7729778, -6.6130686, 22.3832588, -27.7894058, 25.3860474
1: -8.9430733, 19.1648998, -10.8148479, 22.8833485, -31.8264217, 29.9797440
2: -7.2751994, 20.7168579, -8.8901510, 24.6683903, -31.9435902, 29.6070099
3: -7.9728169, 28.5126152, -9.6036949, 33.8211365, -41.7939529, 38.1163101
4: -6.4483242, 26.9256763, -7.8671532, 32.0281372, -38.4764633, 34.7928314

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_B2_A1_A1

### Relational analysis result of NS_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -26.7819156, upper bound: 27.7954689
time: 0.47 seconds

## Relational analysis of NS_B2_A1_A2

### Relational analysis result of NS_B2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -26.4763171, upper bound: 26.7744889
time: 0.49 seconds

## BFS NS instance: NS_B2_A2

### Backsubstitution after applying NS history:
0: -6.6130686, 22.3832588, -6.6130686, 22.3832588, -28.9963264, 28.9963264
1: -10.8148479, 22.8833485, -10.8148479, 22.8833485, -33.6981926, 33.6981926
2: -8.8901510, 24.6683903, -8.8901510, 24.6683903, -33.5585403, 33.5585403
3: -9.6036949, 33.8211365, -9.6036949, 33.8211365, -43.4248314, 43.4248314
4: -7.8671532, 32.0281372, -7.8671532, 32.0281372, -39.8952904, 39.8952904

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B2_A2_A1

### Relational analysis result of NS_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8121678, upper bound: 27.8192465
time: 0.87 seconds

## Relational analysis of NS_B2_A2_A2

### Relational analysis result of NS_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8007885, upper bound: 27.8105578
time: 0.71 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.30 seconds
NS_B1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.30
Output dim: 0, lower bound: -27.7886400, upper bound: 27.7928469
NS_B1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.30
Output dim: 0, lower bound: -27.7931172, upper bound: 27.7975445
NS_B2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 2.30
Output dim: 0, lower bound: -26.7819156, upper bound: 27.7954689
NS_B2_A1_A2, status: Status.VERIFIED, split count: 3, time: 2.30
Output dim: 0, lower bound: -26.4763171, upper bound: 26.7744889
NS_B2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 2.30
Output dim: 0, lower bound: -27.8121678, upper bound: 27.8192465
NS_B2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 2.30
Output dim: 0, lower bound: -27.8007885, upper bound: 27.8105578

## BFS NS instance: NS_B1_B1_A1

### Backsubstitution after applying NS history:
0: -5.4390559, 18.7535896, -4.9898529, 17.4711838, -22.9102402, 23.7434406
1: -8.9748487, 19.1103210, -8.2712097, 17.8242245, -26.7990723, 27.3815308
2: -7.3092508, 20.7006149, -6.7083406, 19.2912483, -26.6004982, 27.4089546
3: -7.9702749, 28.4686832, -7.3757353, 26.5473442, -34.5176086, 35.8444176
4: -6.4465547, 26.8258591, -5.9454603, 25.0227203, -31.4692688, 32.7713089

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B1_B1_A1_B1

### Relational analysis result of NS_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7886400, upper bound: 27.7928469
time: 0.51 seconds

## Relational analysis of NS_B1_B1_A1_B2

### Relational analysis result of NS_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7886400, upper bound: 27.7928469
time: 0.75 seconds

## BFS NS instance: NS_B1_B1_A2

### Backsubstitution after applying NS history:
0: -5.9523649, 19.8514175, -4.4972539, 15.8592100, -21.8115730, 24.3486710
1: -9.7427559, 20.2442093, -7.4725752, 16.1442623, -25.8870182, 27.7167854
2: -7.9848485, 21.9008713, -6.0379863, 17.5366611, -25.5215073, 27.9388561
3: -8.6193991, 30.0439262, -6.6580534, 24.1153564, -32.7347565, 36.7019806
4: -7.0281725, 28.3718929, -5.3449936, 22.6688709, -29.6970444, 33.7168846

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B1_B1_A2_B1

### Relational analysis result of NS_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7931172, upper bound: 27.7975445
time: 0.47 seconds

## Relational analysis of NS_B1_B1_A2_B2

### Relational analysis result of NS_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7931172, upper bound: 27.7975445
time: 0.45 seconds

## BFS NS instance: NS_B2_A1_A1

### Backsubstitution after applying NS history:
0: -4.9898529, 17.4711838, -6.3943958, 21.7289772, -26.7188301, 23.8655796
1: -8.2712097, 17.8242245, -10.4707136, 22.2060680, -30.4772778, 28.2949371
2: -6.7083406, 19.2912483, -8.5976973, 23.9537163, -30.6620560, 27.8889465
3: -7.3757353, 26.5473442, -9.2998905, 32.8476410, -40.2233772, 35.8472328
4: -5.9454603, 25.0227203, -7.6094475, 31.0859413, -37.0313950, 32.6321678

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B2_A1_A1_B1

### Relational analysis result of NS_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7928468, upper bound: 27.7886399
time: 0.60 seconds

## Relational analysis of NS_B2_A1_A1_B2

### Relational analysis result of NS_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7975444, upper bound: 27.7931172
time: 0.87 seconds

## BFS NS instance: NS_B2_A2_A1

### Backsubstitution after applying NS history:
0: -6.0237927, 20.4869289, -6.6130686, 22.3832588, -28.4070511, 27.0999947
1: -9.8778572, 20.8926048, -10.8148479, 22.8833485, -32.7612038, 31.7074528
2: -8.0902863, 22.6051788, -8.8901510, 24.6683903, -32.7586746, 31.4953251
3: -8.7617722, 30.9990101, -9.6036949, 33.8211365, -42.5829048, 40.6026955
4: -7.1387262, 29.2856445, -7.8671532, 32.0281372, -39.1668625, 37.1527977

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_B2_A2_A1_A1

### Relational analysis result of NS_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8124767, upper bound: 27.8128074
time: 0.72 seconds

## Relational analysis of NS_B2_A2_A1_A2

### Relational analysis result of NS_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8115347, upper bound: 27.8117325
time: 0.68 seconds

## BFS NS instance: NS_B2_A2_A2

### Backsubstitution after applying NS history:
0: -6.4581923, 21.4587479, -5.9987688, 20.4320488, -26.8902416, 27.4575138
1: -10.5395784, 21.9054966, -9.8334703, 20.8455296, -31.3851089, 31.7389660
2: -8.6694202, 23.6507874, -8.0649691, 22.5398884, -31.2093086, 31.7157555
3: -9.3299255, 32.3978157, -8.7249622, 30.8982620, -40.2281837, 41.1227722
4: -7.6484089, 30.6516914, -7.1332288, 29.2126102, -36.8610191, 37.7849197

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B2_A2_A2_B1

### Relational analysis result of NS_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8105578, upper bound: 27.8105578
time: 0.67 seconds

## Relational analysis of NS_B2_A2_A2_B2

### Relational analysis result of NS_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8105578, upper bound: 27.8105578
time: 0.58 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 1.98 seconds
NS_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 1.98
Output dim: 0, lower bound: -27.7886400, upper bound: 27.7928469
NS_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 1.98
Output dim: 0, lower bound: -27.7886400, upper bound: 27.7928469
NS_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 1.98
Output dim: 0, lower bound: -27.7931172, upper bound: 27.7975445
NS_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1.98
Output dim: 0, lower bound: -27.7931172, upper bound: 27.7975445
NS_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 1.98
Output dim: 0, lower bound: -27.7928468, upper bound: 27.7886399
NS_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 1.98
Output dim: 0, lower bound: -27.7975444, upper bound: 27.7931172
NS_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 1.98
Output dim: 0, lower bound: -27.8124767, upper bound: 27.8128074
NS_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 1.98
Output dim: 0, lower bound: -27.8115347, upper bound: 27.8117325
NS_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 1.98
Output dim: 0, lower bound: -27.8105578, upper bound: 27.8105578
NS_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1.98
Output dim: 0, lower bound: -27.8105578, upper bound: 27.8105578

## BFS NS instance: NS_B1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -5.4390559, 18.7535896, -4.4955974, 15.8341436, -21.2731972, 23.2491875
1: -8.9748487, 19.1103210, -7.4680882, 16.1045341, -25.0793839, 26.5784092
2: -7.3092508, 20.7006149, -6.0293436, 17.5095921, -24.8188419, 26.7299576
3: -7.9702749, 28.4686832, -6.6484203, 24.0976467, -32.0679207, 35.1171036
4: -6.4465547, 26.8258591, -5.3203859, 22.6204567, -29.0670109, 32.1462440

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_B1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_B1_B1_A1_B1_A1

### Relational analysis result of NS_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7886400, upper bound: 27.7928469
time: 0.69 seconds

## Relational analysis of NS_B1_B1_A1_B1_A2

### Relational analysis result of NS_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7886400, upper bound: 27.7928469
time: 0.51 seconds

## BFS NS instance: NS_B1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -5.4390559, 18.7535896, -4.7342229, 16.4243813, -21.8634338, 23.4878120
1: -8.9748487, 19.1103210, -7.8439040, 16.6983929, -25.6732407, 26.9542255
2: -7.3092508, 20.7006149, -6.3474259, 18.1358376, -25.4450874, 27.0480404
3: -7.9702749, 28.4686832, -6.9785309, 24.9128876, -32.8831520, 35.4472122
4: -6.4465547, 26.8258591, -5.5898933, 23.4206448, -29.8671970, 32.4157486

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_B1_B1_A1_B2_A1

### Relational analysis result of NS_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7886400, upper bound: 27.7928469
time: 0.58 seconds

## Relational analysis of NS_B1_B1_A1_B2_A2

### Relational analysis result of NS_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7886400, upper bound: 27.7928469
time: 0.59 seconds

## BFS NS instance: NS_B1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -5.9523649, 19.8514175, -4.4955974, 15.8341436, -21.7865047, 24.3470154
1: -9.7427559, 20.2442093, -7.4680882, 16.1045341, -25.8472900, 27.7122974
2: -7.9848485, 21.9008713, -6.0293436, 17.5095921, -25.4944363, 27.9302139
3: -8.6193991, 30.0439262, -6.6484203, 24.0976467, -32.7170448, 36.6923447
4: -7.0281725, 28.3718929, -5.3203859, 22.6204567, -29.6486282, 33.6922760

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_B1_B1_A2_B1_A1

### Relational analysis result of NS_B1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7931172, upper bound: 27.7975445
time: 0.52 seconds

## Relational analysis of NS_B1_B1_A2_B1_A2

### Relational analysis result of NS_B1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7931172, upper bound: 27.7975445
time: 0.77 seconds

## BFS NS instance: NS_B1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -5.9523649, 19.8514175, -4.7342229, 16.4243813, -22.3767433, 24.5856400
1: -9.7427559, 20.2442093, -7.8439040, 16.6983929, -26.4411488, 28.0881138
2: -7.9848485, 21.9008713, -6.3474259, 18.1358376, -26.1206818, 28.2482967
3: -8.6193991, 30.0439262, -6.9785309, 24.9128876, -33.5322876, 37.0224571
4: -7.0281725, 28.3718929, -5.5898933, 23.4206448, -30.4488182, 33.9617844

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_B1_B1_A2_B2_A1

### Relational analysis result of NS_B1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7931172, upper bound: 27.7975445
time: 0.52 seconds

## Relational analysis of NS_B1_B1_A2_B2_A2

### Relational analysis result of NS_B1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7931172, upper bound: 27.7975445
time: 0.83 seconds

## BFS NS instance: NS_B2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -4.9898529, 17.4711838, -5.8092694, 19.8465462, -24.8363991, 23.2804527
1: -8.2712097, 17.8242245, -9.5397997, 20.2309399, -28.5021496, 27.3640175
2: -6.7083406, 19.2912483, -7.8035226, 21.9050274, -28.6133690, 27.0947704
3: -7.3757353, 26.5473442, -8.4638977, 30.0456333, -37.4213676, 35.0112343
4: -5.9454603, 25.0227203, -6.8858380, 28.3614311, -34.3068886, 31.9085579

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B2_A1_A1_B1_A1

### Relational analysis result of NS_B2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7928468, upper bound: 27.7886399
time: 0.63 seconds

## Relational analysis of NS_B2_A1_A1_B1_A2

### Relational analysis result of NS_B2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7928468, upper bound: 27.7886400
time: 0.72 seconds

## BFS NS instance: NS_B2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -4.4972539, 15.8592100, -6.2166338, 20.6962109, -25.1934643, 22.0758438
1: -7.4725752, 16.1442623, -10.1557188, 21.1222668, -28.5948410, 26.2999763
2: -6.0379863, 17.5366611, -8.3447580, 22.8237419, -28.8617249, 25.8814163
3: -6.6580534, 24.1153564, -8.9890356, 31.2627144, -37.9207687, 33.1043930
4: -5.3449936, 22.6688709, -7.3607793, 29.5540562, -34.8990479, 30.0296497

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B2_A1_A1_B2_A1

### Relational analysis result of NS_B2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7975444, upper bound: 27.7931172
time: 0.56 seconds

## Relational analysis of NS_B2_A1_A1_B2_A2

### Relational analysis result of NS_B2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7975444, upper bound: 27.7931172
time: 0.53 seconds

## BFS NS instance: NS_B2_A2_A1_A1

### Backsubstitution after applying NS history:
0: -5.4029026, 18.6289616, -6.3943958, 21.7289772, -27.1318798, 25.0233574
1: -8.9004507, 18.9769077, -10.4707136, 22.2060680, -31.1065178, 29.4476204
2: -7.2595968, 20.5732021, -8.5976973, 23.9537163, -31.2133141, 29.1708984
3: -7.9015369, 28.2320576, -9.2998905, 32.8476410, -40.7491760, 37.5319481
4: -6.4061265, 26.6033630, -7.6094475, 31.0859413, -37.4920616, 34.2128105

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B2_A2_A1_A1_B1

### Relational analysis result of NS_B2_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8124767, upper bound: 27.8128074
time: 0.52 seconds

## Relational analysis of NS_B2_A2_A1_A1_B2

### Relational analysis result of NS_B2_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8124767, upper bound: 27.8128074
time: 0.52 seconds

## BFS NS instance: NS_B2_A2_A1_A2

### Backsubstitution after applying NS history:
0: -7.8388243, 25.1275501, -6.1698418, 20.9670200, -28.8058414, 31.2973919
1: -12.6399717, 25.7362232, -10.0991879, 21.4146290, -34.0545998, 35.8354111
2: -10.4622936, 27.6880188, -8.2906551, 23.1323757, -33.5946693, 35.9786758
3: -11.1317329, 37.9511490, -8.9571447, 31.6838322, -42.8155670, 46.9082947
4: -9.2031116, 35.9791756, -7.3312769, 29.9680061, -39.1711197, 43.3104477

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B2_A2_A1_A2_B1

### Relational analysis result of NS_B2_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8109195, upper bound: 27.8114979
time: 0.69 seconds

## Relational analysis of NS_B2_A2_A1_A2_B2

### Relational analysis result of NS_B2_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8109195, upper bound: 27.8117325
time: 0.57 seconds

## BFS NS instance: NS_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -6.4581923, 21.4587479, -6.0193100, 20.4722252, -26.9304180, 27.4780560
1: -10.5395784, 21.9054966, -9.8706770, 20.8774643, -31.4170380, 31.7761726
2: -8.6694202, 23.6507874, -8.0841484, 22.5891762, -31.2585964, 31.7349358
3: -9.3299255, 32.3978157, -8.7553902, 30.9768677, -40.3067856, 41.1532021
4: -7.6484089, 30.6516914, -7.1333265, 29.2644329, -36.9128418, 37.7850189

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_B2_A2_A2_B1_A1

### Relational analysis result of NS_B2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8087354, upper bound: 27.8087354
time: 0.65 seconds

## Relational analysis of NS_B2_A2_A2_B1_A2

### Relational analysis result of NS_B2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8105035, upper bound: 27.8105035
time: 0.67 seconds

## BFS NS instance: NS_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -6.4581923, 21.4587479, -6.4581923, 21.4587479, -27.9169407, 27.9169407
1: -10.5395784, 21.9054966, -10.5395784, 21.9054966, -32.4450760, 32.4450760
2: -8.6694202, 23.6507874, -8.6694202, 23.6507874, -32.3202057, 32.3202057
3: -9.3299255, 32.3978157, -9.3299255, 32.3978157, -41.7277412, 41.7277412
4: -7.6484089, 30.6516914, -7.6484089, 30.6516914, -38.3001022, 38.3001022

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B2_A2_A2_B2_B1

### Relational analysis result of NS_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8092438, upper bound: 27.8099951
time: 0.80 seconds

## Relational analysis of NS_B2_A2_A2_B2_B2

### Relational analysis result of NS_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8105035, upper bound: 27.8105035
time: 0.62 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 2.18 seconds
NS_B1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.18
Output dim: 0, lower bound: -27.7886400, upper bound: 27.7928469
NS_B1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.18
Output dim: 0, lower bound: -27.7886400, upper bound: 27.7928469
NS_B1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.18
Output dim: 0, lower bound: -27.7886400, upper bound: 27.7928469
NS_B1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.18
Output dim: 0, lower bound: -27.7886400, upper bound: 27.7928469
NS_B1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.18
Output dim: 0, lower bound: -27.7931172, upper bound: 27.7975445
NS_B1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.18
Output dim: 0, lower bound: -27.7931172, upper bound: 27.7975445
NS_B1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.18
Output dim: 0, lower bound: -27.7931172, upper bound: 27.7975445
NS_B1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.18
Output dim: 0, lower bound: -27.7931172, upper bound: 27.7975445
NS_B2_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.18
Output dim: 0, lower bound: -27.7928468, upper bound: 27.7886399
NS_B2_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.18
Output dim: 0, lower bound: -27.7928468, upper bound: 27.7886400
NS_B2_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.18
Output dim: 0, lower bound: -27.7975444, upper bound: 27.7931172
NS_B2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.18
Output dim: 0, lower bound: -27.7975444, upper bound: 27.7931172
NS_B2_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.18
Output dim: 0, lower bound: -27.8124767, upper bound: 27.8128074
NS_B2_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.18
Output dim: 0, lower bound: -27.8124767, upper bound: 27.8128074
NS_B2_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.18
Output dim: 0, lower bound: -27.8109195, upper bound: 27.8114979
NS_B2_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.18
Output dim: 0, lower bound: -27.8109195, upper bound: 27.8117325
NS_B2_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.18
Output dim: 0, lower bound: -27.8087354, upper bound: 27.8087354
NS_B2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.18
Output dim: 0, lower bound: -27.8105035, upper bound: 27.8105035
NS_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.18
Output dim: 0, lower bound: -27.8092438, upper bound: 27.8099951
NS_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.18
Output dim: 0, lower bound: -27.8105035, upper bound: 27.8105035

## BFS NS instance: NS_B1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -5.1356249, 17.8190632, -4.4955974, 15.8341436, -20.9697666, 22.3146610
1: -8.4906330, 18.1508827, -7.4680882, 16.1045341, -24.5951672, 25.6189709
2: -6.9010758, 19.6838284, -6.0293436, 17.5095921, -24.4106655, 25.7131710
3: -7.5423803, 27.0686264, -6.6484203, 24.0976467, -31.6400261, 33.7170486
4: -6.0849333, 25.4665070, -5.3203859, 22.6204567, -28.7053909, 30.7868919

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B1_B1_A1_B1_A1_B1

### Relational analysis result of NS_B1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -26.8224753, upper bound: 26.8591658
time: 0.78 seconds

## Relational analysis of NS_B1_B1_A1_B1_A1_B2

### Relational analysis result of NS_B1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -26.6297176, upper bound: 26.6317381
time: 0.75 seconds

## BFS NS instance: NS_B1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -7.7255750, 24.7534657, -4.4955974, 15.8341436, -23.5597153, 29.2490635
1: -12.4643726, 25.3330688, -7.4680882, 16.1045341, -28.5689049, 32.8011551
2: -10.3146629, 27.2893505, -6.0293436, 17.5095921, -27.8242550, 33.3186798
3: -10.9729271, 37.4260025, -6.6484203, 24.0976467, -35.0705719, 44.0744247
4: -9.0796299, 35.4806595, -5.3203859, 22.6204567, -31.7000866, 40.8010445

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B1_B1_A1_B1_A2_A1

### Relational analysis result of NS_B1_B1_A1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.0313846, upper bound: 26.9792314
time: 0.96 seconds

## Relational analysis of NS_B1_B1_A1_B1_A2_A2

### Relational analysis result of NS_B1_B1_A1_B1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -26.6297176, upper bound: 26.6317381
time: 0.50 seconds

## BFS NS instance: NS_B1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -5.1356249, 17.8190632, -4.7342229, 16.4243813, -21.5600033, 22.5532856
1: -8.4906330, 18.1508827, -7.8439040, 16.6983929, -25.1890240, 25.9947872
2: -6.9010758, 19.6838284, -6.3474259, 18.1358376, -25.0369110, 26.0312538
3: -7.5423803, 27.0686264, -6.9785309, 24.9128876, -32.4552612, 34.0471573
4: -6.0849333, 25.4665070, -5.5898933, 23.4206448, -29.5055771, 31.0564003

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_B1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B1_B1_A1_B2_A1_A1

### Relational analysis result of NS_B1_B1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7683350, upper bound: 27.7724263
time: 1.24 seconds

## Relational analysis of NS_B1_B1_A1_B2_A1_A2

### Relational analysis result of NS_B1_B1_A1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7295923, upper bound: 27.7202259
time: 0.59 seconds

## BFS NS instance: NS_B1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -7.7255750, 24.7534657, -4.7342229, 16.4243813, -24.1499500, 29.4876881
1: -12.4643726, 25.3330688, -7.8439040, 16.6983929, -29.1627617, 33.1769714
2: -10.3146629, 27.2893505, -6.3474259, 18.1358376, -28.4505005, 33.6367683
3: -10.9729271, 37.4260025, -6.9785309, 24.9128876, -35.8858147, 44.4045296
4: -9.0796299, 35.4806595, -5.5898933, 23.4206448, -32.5002747, 41.0705528

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_B1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B1_B1_A1_B2_A2_A1

### Relational analysis result of NS_B1_B1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7683350, upper bound: 27.7724263
time: 0.64 seconds

## Relational analysis of NS_B1_B1_A1_B2_A2_A2

### Relational analysis result of NS_B1_B1_A1_B2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7295923, upper bound: 27.7202259
time: 0.48 seconds

## BFS NS instance: NS_B1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -5.7053041, 19.0779877, -4.4955974, 15.8341436, -21.5394478, 23.5735855
1: -9.3465805, 19.4463367, -7.4680882, 16.1045341, -25.4511147, 26.9144249
2: -7.6509905, 21.0604229, -6.0293436, 17.5095921, -25.1605816, 27.0897655
3: -8.2686615, 28.8807220, -6.6484203, 24.0976467, -32.3663101, 35.5291367
4: -6.7312346, 27.2469158, -5.3203859, 22.6204567, -29.3516922, 32.5673027

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B1_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B1_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B1_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_B1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B1_B1_A2_B1_A1_B1

### Relational analysis result of NS_B1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7818536, upper bound: 27.7801500
time: 0.46 seconds

## Relational analysis of NS_B1_B1_A2_B1_A1_B2

### Relational analysis result of NS_B1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7682409, upper bound: 27.7685098
time: 0.53 seconds

## BFS NS instance: NS_B1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -7.6382642, 23.9475174, -4.4955974, 15.8341436, -23.4724045, 28.4431152
1: -12.2790642, 24.5372181, -7.4680882, 16.1045341, -28.3835983, 32.0052986
2: -10.1710396, 26.3765144, -6.0293436, 17.5095921, -27.6806278, 32.4058495
3: -10.7825270, 36.1794167, -6.6484203, 24.0976467, -34.8801727, 42.8278351
4: -8.9195261, 34.3002892, -5.3203859, 22.6204567, -31.5399799, 39.6206741

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B1_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B1_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B1_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_B1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B1_B1_A2_B1_A2_B1

### Relational analysis result of NS_B1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7818536, upper bound: 27.7801500
time: 0.53 seconds

## Relational analysis of NS_B1_B1_A2_B1_A2_B2

### Relational analysis result of NS_B1_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7682409, upper bound: 27.7685098
time: 0.52 seconds

## BFS NS instance: NS_B1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -5.7053041, 19.0779877, -4.7342229, 16.4243813, -22.1296844, 23.8122101
1: -9.3465805, 19.4463367, -7.8439040, 16.6983929, -26.0449734, 27.2902412
2: -7.6509905, 21.0604229, -6.3474259, 18.1358376, -25.7868271, 27.4078484
3: -8.2686615, 28.8807220, -6.9785309, 24.9128876, -33.1815491, 35.8592453
4: -6.7312346, 27.2469158, -5.5898933, 23.4206448, -30.1518784, 32.8368073

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_B1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B1_B1_A2_B2_A1_B1

### Relational analysis result of NS_B1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7807591, upper bound: 27.7790637
time: 0.85 seconds

## Relational analysis of NS_B1_B1_A2_B2_A1_B2

### Relational analysis result of NS_B1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7751234, upper bound: 27.7739506
time: 0.88 seconds

## BFS NS instance: NS_B1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -7.6382642, 23.9475174, -4.7342229, 16.4243813, -24.0626431, 28.6817398
1: -12.2790642, 24.5372181, -7.8439040, 16.6983929, -28.9774570, 32.3811188
2: -10.1710396, 26.3765144, -6.3474259, 18.1358376, -28.3068771, 32.7239418
3: -10.7825270, 36.1794167, -6.9785309, 24.9128876, -35.6954155, 43.1579437
4: -8.9195261, 34.3002892, -5.5898933, 23.4206448, -32.3401718, 39.8901825

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_B1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B1_B1_A2_B2_A2_A1

### Relational analysis result of NS_B1_B1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7681902, upper bound: 27.7863296
time: 0.79 seconds

## Relational analysis of NS_B1_B1_A2_B2_A2_A2

### Relational analysis result of NS_B1_B1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7751234, upper bound: 27.7739506
time: 0.84 seconds

## BFS NS instance: NS_B2_A1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -4.4955974, 15.8341436, -5.8092694, 19.8465462, -24.3421440, 21.6434116
1: -7.4680882, 16.1045341, -9.5397997, 20.2309399, -27.6990280, 25.6443272
2: -6.0293436, 17.5095921, -7.8035226, 21.9050274, -27.9343719, 25.3131123
3: -6.6484203, 24.0976467, -8.4638977, 30.0456333, -36.6940536, 32.5615463
4: -5.3203859, 22.6204567, -6.8858380, 28.3614311, -33.6818161, 29.5062943

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B2_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_B2_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B2_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B2_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B2_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B2_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_B2_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B2_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B2_A1_A1_B1_A1_A1

### Relational analysis result of NS_B2_A1_A1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7252887, upper bound: 27.7351598
time: 0.74 seconds

## Relational analysis of NS_B2_A1_A1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 17

## BFS NS instance: NS_B2_A1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -4.7342229, 16.4243813, -5.8092694, 19.8465462, -24.5807686, 22.2336483
1: -7.8439040, 16.6983929, -9.5397997, 20.2309399, -28.0748444, 26.2381840
2: -6.3474259, 18.1358376, -7.8035226, 21.9050274, -28.2524529, 25.9393597
3: -6.9785309, 24.9128876, -8.4638977, 30.0456333, -37.0241623, 33.3767815
4: -5.5898933, 23.4206448, -6.8858380, 28.3614311, -33.9513245, 30.3064823

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_B2_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B2_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B2_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B2_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B2_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B2_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B2_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B2_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B2_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 9

## BFS NS instance: NS_B2_A1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -4.4955974, 15.8341436, -6.2166338, 20.6962109, -25.1918087, 22.0507755
1: -7.4680882, 16.1045341, -10.1557188, 21.1222668, -28.5903549, 26.2602463
2: -6.0293436, 17.5095921, -8.3447580, 22.8237419, -28.8530846, 25.8543491
3: -6.6484203, 24.0976467, -8.9890356, 31.2627144, -37.9111328, 33.0866814
4: -5.3203859, 22.6204567, -7.3607793, 29.5540562, -34.8744431, 29.9812355

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_B2_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B2_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B2_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B2_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B2_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B2_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B2_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B2_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B2_A1_A1_B2_A1_A1

### Relational analysis result of NS_B2_A1_A1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7252887, upper bound: 27.7351598
time: 0.63 seconds

## Relational analysis of NS_B2_A1_A1_B2_A1_A2

### Relational analysis result of NS_B2_A1_A1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7141902, upper bound: 27.7682409
time: 0.50 seconds

## BFS NS instance: NS_B2_A1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -4.7342229, 16.4243813, -6.2166338, 20.6962109, -25.4304333, 22.6410141
1: -7.8439040, 16.6983929, -10.1557188, 21.1222668, -28.9661713, 26.8541069
2: -6.3474259, 18.1358376, -8.3447580, 22.8237419, -29.1711674, 26.4805946
3: -6.9785309, 24.9128876, -8.9890356, 31.2627144, -38.2412415, 33.9019241
4: -5.5898933, 23.4206448, -7.3607793, 29.5540562, -35.1439476, 30.7814236

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B2_A1_A1_B2_A2_B1

### Relational analysis result of NS_B2_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7928468, upper bound: 27.7891798
time: 1.15 seconds

## Relational analysis of NS_B2_A1_A1_B2_A2_B2

### Relational analysis result of NS_B2_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7928468, upper bound: 27.7891798
time: 0.48 seconds

## BFS NS instance: NS_B2_A2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -5.4029026, 18.6289616, -5.8092694, 19.8465462, -25.2494450, 24.4382305
1: -8.9004507, 18.9769077, -9.5397997, 20.2309399, -29.1313896, 28.5166988
2: -7.2595968, 20.5732021, -7.8035226, 21.9050274, -29.1646233, 28.3767242
3: -7.9015369, 28.2320576, -8.4638977, 30.0456333, -37.9471703, 36.6959496
4: -6.4061265, 26.6033630, -6.8858380, 28.3614311, -34.7675552, 33.4892006

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B2_A2_A1_A1_B1_B1

### Relational analysis result of NS_B2_A2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8124767, upper bound: 27.8128073
time: 0.61 seconds

## Relational analysis of NS_B2_A2_A1_A1_B1_B2

### Relational analysis result of NS_B2_A2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8124767, upper bound: 27.8128073
time: 0.53 seconds

## BFS NS instance: NS_B2_A2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -5.4029026, 18.6289616, -6.2166338, 20.6962109, -26.0991135, 24.8455963
1: -8.9004507, 18.9769077, -10.1557188, 21.1222668, -30.0227165, 29.1326237
2: -7.2595968, 20.5732021, -8.3447580, 22.8237419, -30.0833397, 28.9179592
3: -7.9015369, 28.2320576, -8.9890356, 31.2627144, -39.1642532, 37.2210922
4: -6.4061265, 26.6033630, -7.3607793, 29.5540562, -35.9601784, 33.9641380

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_A2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B2_A2_A1_A1_B2_B1

### Relational analysis result of NS_B2_A2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8124767, upper bound: 27.8128074
time: 1.13 seconds

## Relational analysis of NS_B2_A2_A1_A1_B2_B2

### Relational analysis result of NS_B2_A2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8124767, upper bound: 27.8128073
time: 0.54 seconds

## BFS NS instance: NS_B2_A2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -7.8388243, 25.1275501, -5.9776387, 20.4788971, -28.3177185, 31.1051884
1: -12.6399717, 25.7362232, -9.8153677, 20.9132309, -33.5531921, 35.5515900
2: -10.4622936, 27.6880188, -8.0404854, 22.5888958, -33.0511818, 35.7285042
3: -11.1317329, 37.9511490, -8.7233858, 30.9885597, -42.1202850, 46.6745338
4: -9.2031116, 35.9791756, -7.1184216, 29.2894173, -38.4925308, 43.0975952

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_B2_A2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B2_A2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B2_A2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_B2_A2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B2_A2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B2_A2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B2_A2_A1_A2_B1_B1

### Relational analysis result of NS_B2_A2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8109195, upper bound: 27.8114979
time: 0.71 seconds

## Relational analysis of NS_B2_A2_A1_A2_B1_B2

### Relational analysis result of NS_B2_A2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8109195, upper bound: 27.8114979
time: 0.84 seconds

## BFS NS instance: NS_B2_A2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -7.8388243, 25.1275501, -8.5507870, 27.4290867, -35.2679100, 33.6783371
1: -12.6399717, 25.7362232, -13.7664528, 28.1737900, -40.8137627, 39.5026779
2: -10.4622936, 27.6880188, -11.4451046, 30.1721191, -40.6344147, 39.1331253
3: -11.1317329, 37.9511490, -12.1364708, 41.3485565, -52.4802856, 50.0876198
4: -9.2031116, 35.9791756, -10.0589399, 39.1992722, -48.4023819, 46.0381088

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_B2_A2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B2_A2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B2_A2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B2_A2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B2_A2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B2_A2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B2_A2_A1_A2_B2_B1

### Relational analysis result of NS_B2_A2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8109195, upper bound: 27.8114979
time: 0.66 seconds

## Relational analysis of NS_B2_A2_A1_A2_B2_B2

### Relational analysis result of NS_B2_A2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8109195, upper bound: 27.8114979
time: 1.21 seconds

## BFS NS instance: NS_B2_A2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -5.7951403, 19.3845100, -5.8048992, 19.8322334, -25.6273670, 25.1894093
1: -9.4860306, 19.7728939, -9.5328035, 20.2161884, -29.7022190, 29.3056984
2: -7.7768764, 21.4003086, -7.7975368, 21.8894577, -29.6663342, 29.1978455
3: -8.3971539, 29.3123341, -8.4576797, 30.0240822, -38.4212265, 37.7700119
4: -6.8590794, 27.6703186, -6.8805737, 28.3408165, -35.1998901, 34.5508919

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B2_A2_A2_B1_A1_B1

### Relational analysis result of NS_B2_A2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8087353, upper bound: 27.8087354
time: 0.91 seconds

## Relational analysis of NS_B2_A2_A2_B1_A1_B2

### Relational analysis result of NS_B2_A2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8087353, upper bound: 27.8087354
time: 0.61 seconds

## BFS NS instance: NS_B2_A2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -7.6980271, 24.1949196, -5.6223931, 19.2334023, -26.9314289, 29.8173084
1: -12.3796749, 24.7970486, -9.2341299, 19.6000271, -31.9796925, 34.0311775
2: -10.2556095, 26.6462612, -7.5476580, 21.2427998, -31.4984093, 34.1939163
3: -10.8816719, 36.5377197, -8.1842995, 29.1116581, -39.9933319, 44.7220192
4: -8.9968672, 34.6487350, -6.6634550, 27.4741802, -36.4710464, 41.3121910

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B2_A2_A2_B1_A2_B1

### Relational analysis result of NS_B2_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8120063, upper bound: 27.8121792
time: 0.61 seconds

## Relational analysis of NS_B2_A2_A2_B1_A2_B2

### Relational analysis result of NS_B2_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8120063, upper bound: 27.8121792
time: 0.63 seconds

## BFS NS instance: NS_B2_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -6.2166338, 20.6962109, -5.7951403, 19.3845100, -25.6011429, 26.4913483
1: -10.1557188, 21.1222668, -9.4860306, 19.7728939, -29.9286118, 30.6082954
2: -8.3447580, 22.8237419, -7.7768764, 21.4003086, -29.7450676, 30.6006184
3: -8.9890356, 31.2627144, -8.3971539, 29.3123341, -38.3013687, 39.6598625
4: -7.3607793, 29.5540562, -6.8590794, 27.6703186, -35.0310898, 36.4131355

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_B2_A2_A2_B2_B1_A1

### Relational analysis result of NS_B2_A2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8087354, upper bound: 27.8087354
time: 0.78 seconds

## Relational analysis of NS_B2_A2_A2_B2_B1_A2

### Relational analysis result of NS_B2_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8087354, upper bound: 27.8099951
time: 0.67 seconds

## BFS NS instance: NS_B2_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -6.1007204, 20.3800793, -7.6980271, 24.1949196, -30.2956390, 28.0781059
1: -9.9659195, 20.8025322, -12.3796749, 24.7970486, -34.7629700, 33.1822052
2: -8.1853418, 22.4749603, -10.2556095, 26.6462612, -34.8316040, 32.7305679
3: -8.8210478, 30.7763462, -10.8816719, 36.5377197, -45.3587685, 41.6580200
4: -7.2291818, 29.1014824, -8.9968672, 34.6487350, -41.8779182, 38.0983467

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_B2_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B2_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B2_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B2_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B2_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_B2_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B2_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B2_A2_A2_B2_B2_B1

### Relational analysis result of NS_B2_A2_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7674769, upper bound: 27.7648134
time: 0.60 seconds

## Relational analysis of NS_B2_A2_A2_B2_B2_B2

### Relational analysis result of NS_B2_A2_A2_B2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.69 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.86 seconds
NS_B1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.86
Output dim: 0, lower bound: -26.8224753, upper bound: 26.8591658
NS_B1_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.86
Output dim: 0, lower bound: -26.6297176, upper bound: 26.6317381
NS_B1_B1_A1_B1_A2_A1, status: Status.VERIFIED, split count: 6, time: 3.86
Output dim: 0, lower bound: -27.0313846, upper bound: 26.9792314
NS_B1_B1_A1_B1_A2_A2, status: Status.VERIFIED, split count: 6, time: 3.86
Output dim: 0, lower bound: -26.6297176, upper bound: 26.6317381
NS_B1_B1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -27.7683350, upper bound: 27.7724263
NS_B1_B1_A1_B2_A1_A2, status: Status.VERIFIED, split count: 6, time: 3.86
Output dim: 0, lower bound: -27.7295923, upper bound: 27.7202259
NS_B1_B1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -27.7683350, upper bound: 27.7724263
NS_B1_B1_A1_B2_A2_A2, status: Status.VERIFIED, split count: 6, time: 3.86
Output dim: 0, lower bound: -27.7295923, upper bound: 27.7202259
NS_B1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -27.7818536, upper bound: 27.7801500
NS_B1_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.86
Output dim: 0, lower bound: -27.7682409, upper bound: 27.7685098
NS_B1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -27.7818536, upper bound: 27.7801500
NS_B1_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.86
Output dim: 0, lower bound: -27.7682409, upper bound: 27.7685098
NS_B1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -27.7807591, upper bound: 27.7790637
NS_B1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -27.7751234, upper bound: 27.7739506
NS_B1_B1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -27.7681902, upper bound: 27.7863296
NS_B1_B1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -27.7751234, upper bound: 27.7739506
NS_B2_A1_A1_B2_A1_A1, status: Status.VERIFIED, split count: 6, time: 3.86
Output dim: 0, lower bound: -27.7252887, upper bound: 27.7351598
NS_B2_A1_A1_B2_A1_A2, status: Status.VERIFIED, split count: 6, time: 3.86
Output dim: 0, lower bound: -27.7141902, upper bound: 27.7682409
NS_B2_A1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -27.7928468, upper bound: 27.7891798
NS_B2_A1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -27.7928468, upper bound: 27.7891798
NS_B2_A2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -27.8124767, upper bound: 27.8128073
NS_B2_A2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -27.8124767, upper bound: 27.8128073
NS_B2_A2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -27.8124767, upper bound: 27.8128074
NS_B2_A2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -27.8124767, upper bound: 27.8128073
NS_B2_A2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -27.8109195, upper bound: 27.8114979
NS_B2_A2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -27.8109195, upper bound: 27.8114979
NS_B2_A2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -27.8109195, upper bound: 27.8114979
NS_B2_A2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -27.8109195, upper bound: 27.8114979
NS_B2_A2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -27.8087353, upper bound: 27.8087354
NS_B2_A2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -27.8087353, upper bound: 27.8087354
NS_B2_A2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -27.8120063, upper bound: 27.8121792
NS_B2_A2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -27.8120063, upper bound: 27.8121792
NS_B2_A2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -27.8087354, upper bound: 27.8087354
NS_B2_A2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -27.8087354, upper bound: 27.8099951
NS_B2_A2_A2_B2_B2_B1, status: Status.VERIFIED, split count: 6, time: 3.86
Output dim: 0, lower bound: -27.7674769, upper bound: 27.7648134
NS_B2_A2_A2_B2_B2_B2, status: Status.VERIFIED, split count: 6, time: 3.86
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121

## BFS NS instance: NS_B1_B1_A1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -5.0616093, 17.5872383, -4.7342229, 16.4243813, -21.4859905, 22.3214607
1: -8.3727589, 17.9134541, -7.8439040, 16.6983929, -25.0711460, 25.7573586
2: -6.8010635, 19.4322319, -6.3474259, 18.1358376, -24.9368973, 25.7796574
3: -7.4384723, 26.7205925, -6.9785309, 24.9128876, -32.3513527, 33.6991234
4: -5.9991608, 25.1323872, -5.5898933, 23.4206448, -29.4198055, 30.7222805

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B1_B1_A1_B2_A1_A1_B1

### Relational analysis result of NS_B1_B1_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7706916, upper bound: 27.7709281
time: 0.71 seconds

## Relational analysis of NS_B1_B1_A1_B2_A1_A1_B2

### Relational analysis result of NS_B1_B1_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7706916, upper bound: 27.7709281
time: 0.62 seconds

## BFS NS instance: NS_B1_B1_A1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -7.6131754, 24.4086399, -4.7342229, 16.4243813, -24.0375557, 29.1428623
1: -12.2846479, 24.9795895, -7.8439040, 16.6983929, -28.9830379, 32.8234940
2: -10.1630287, 26.9127998, -6.3474259, 18.1358376, -28.2988663, 33.2602272
3: -10.8155365, 36.9066238, -6.9785309, 24.9128876, -35.7284164, 43.8851471
4: -8.9476757, 34.9791336, -5.5898933, 23.4206448, -32.3683205, 40.5690269

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B1_B1_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B1_B1_A1_B2_A2_A1_B1

### Relational analysis result of NS_B1_B1_A1_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7295923, upper bound: 27.7202259
time: 0.51 seconds

## Relational analysis of NS_B1_B1_A1_B2_A2_A1_B2

### Relational analysis result of NS_B1_B1_A1_B2_A2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7295923, upper bound: 27.7202259
time: 0.66 seconds

## BFS NS instance: NS_B1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -5.7053041, 19.0779877, -4.4252939, 15.6111059, -21.3164101, 23.5032787
1: -9.3465805, 19.4463367, -7.3541880, 15.8749866, -25.2215672, 26.8005257
2: -7.6509905, 21.0604229, -5.9332600, 17.2657795, -24.9167709, 26.9936829
3: -8.2686615, 28.8807220, -6.5480304, 23.7588024, -32.0274658, 35.4287376
4: -6.7312346, 27.2469158, -5.2362361, 22.2943745, -29.0256081, 32.4831505

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_B1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7799429, upper bound: 27.7876283
time: 0.99 seconds

## Relational analysis of NS_B1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_B1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7799429, upper bound: 27.7876283
time: 0.58 seconds

## BFS NS instance: NS_B1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -7.6382642, 23.9475174, -4.4252939, 15.6111059, -23.2493687, 28.3728065
1: -12.2790642, 24.5372181, -7.3541880, 15.8749866, -28.1540508, 31.8914051
2: -10.1710396, 26.3765144, -5.9332600, 17.2657795, -27.4368172, 32.3097687
3: -10.7825270, 36.1794167, -6.5480304, 23.7588024, -34.5413284, 42.7274399
4: -8.9195261, 34.3002892, -5.2362361, 22.2943745, -31.2138901, 39.5365257

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_B1_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7682409, upper bound: 27.7685098
time: 0.74 seconds

## Relational analysis of NS_B1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_B1_B1_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7682409, upper bound: 27.7685098
time: 1.17 seconds

## BFS NS instance: NS_B1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -5.7053041, 19.0779877, -4.6568985, 16.1854362, -21.8907394, 23.7348824
1: -9.3465805, 19.4463367, -7.7200298, 16.4539776, -25.8005581, 27.1663666
2: -7.6509905, 21.0604229, -6.2426319, 17.8746071, -25.5255966, 27.3030548
3: -8.2686615, 28.8807220, -6.8696055, 24.5522022, -32.8208618, 35.7503242
4: -6.7312346, 27.2469158, -5.4995651, 23.0713387, -29.8025742, 32.7464828

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_B1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_B1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7706916, upper bound: 27.7844649
time: 0.70 seconds

## Relational analysis of NS_B1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_B1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7868138, upper bound: 27.7844649
time: 0.54 seconds

## BFS NS instance: NS_B1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -5.6486168, 18.8913002, -4.6114683, 15.9813213, -21.6299381, 23.5027676
1: -9.2556572, 19.2573872, -7.6433921, 16.2519264, -25.5075836, 26.9007797
2: -7.5751128, 20.8585777, -6.1802125, 17.6584473, -25.2335606, 27.0387897
3: -8.1888866, 28.5980377, -6.8026519, 24.2351742, -32.4240608, 35.4006882
4: -6.6655602, 26.9770966, -5.4443140, 22.7835884, -29.4491482, 32.4214096

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_B1_B1_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_B1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7706916, upper bound: 27.7844649
time: 0.75 seconds

## Relational analysis of NS_B1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_B1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7868138, upper bound: 27.7844649
time: 0.82 seconds

## BFS NS instance: NS_B1_B1_A2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -7.5200233, 23.5865841, -4.7342229, 16.4243813, -23.9444046, 28.3208065
1: -12.0903320, 24.1655769, -7.8439040, 16.6983929, -28.7887249, 32.0094795
2: -10.0117388, 25.9833679, -6.3474259, 18.1358376, -28.1475754, 32.3307953
3: -10.6170120, 35.6386681, -6.9785309, 24.9128876, -35.5298996, 42.6171989
4: -8.7812548, 33.7779312, -5.5898933, 23.4206448, -32.2018967, 39.3678246

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B1_B1_A2_B2_A2_A1_B1

### Relational analysis result of NS_B1_B1_A2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7751234, upper bound: 27.7739506
time: 0.85 seconds

## Relational analysis of NS_B1_B1_A2_B2_A2_A1_B2

### Relational analysis result of NS_B1_B1_A2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7751234, upper bound: 27.7739506
time: 0.80 seconds

## BFS NS instance: NS_B1_B1_A2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -7.4972787, 23.4641647, -4.6876974, 16.2719917, -23.7692699, 28.1518593
1: -12.0495205, 24.0436764, -7.7687588, 16.5432625, -28.5927811, 31.8124352
2: -9.9798918, 25.8561954, -6.2846880, 17.9702454, -27.9501381, 32.1408844
3: -10.5811272, 35.4473038, -6.9121943, 24.6829185, -35.2640457, 42.3594971
4: -8.7525120, 33.5995064, -5.5351381, 23.2003517, -31.9528599, 39.1346397

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_B1_B1_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B1_B1_A2_B2_A2_A2_B1

### Relational analysis result of NS_B1_B1_A2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7222874, upper bound: 27.7739506
time: 0.81 seconds

## Relational analysis of NS_B1_B1_A2_B2_A2_A2_B2

### Relational analysis result of NS_B1_B1_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7751234, upper bound: 27.7739506
time: 0.63 seconds

## BFS NS instance: NS_B2_A1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -4.7342229, 16.4243813, -5.7951403, 19.3845100, -24.1187325, 22.2195168
1: -7.8439040, 16.6983929, -9.4860306, 19.7728939, -27.6167984, 26.1844196
2: -6.3474259, 18.1358376, -7.7768764, 21.4003086, -27.7477341, 25.9127140
3: -6.9785309, 24.9128876, -8.3971539, 29.3123341, -36.2908630, 33.3100319
4: -5.5898933, 23.4206448, -6.8590794, 27.6703186, -33.2602119, 30.2797241

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B2_A1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_B2_A1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B2_A1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B2_A1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B2_A1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B2_A1_A1_B2_A2_B1_A1

### Relational analysis result of NS_B2_A1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7468375, upper bound: 27.7509987
time: 0.70 seconds

## Relational analysis of NS_B2_A1_A1_B2_A2_B1_A2

### Relational analysis result of NS_B2_A1_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7607956, upper bound: 27.7607880
time: 0.56 seconds

## BFS NS instance: NS_B2_A1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -4.7342229, 16.4243813, -7.6980271, 24.1949196, -28.9291420, 24.1224079
1: -7.8439040, 16.6983929, -12.3796749, 24.7970486, -32.6409531, 29.0780582
2: -6.3474259, 18.1358376, -10.2556095, 26.6462612, -32.9936867, 28.3914471
3: -6.9785309, 24.9128876, -10.8816719, 36.5377197, -43.5162506, 35.7945595
4: -5.5898933, 23.4206448, -8.9968672, 34.6487350, -40.2386284, 32.4175110

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B2_A1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_B2_A1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B2_A1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B2_A1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B2_A1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B2_A1_A1_B2_A2_B2_B1

### Relational analysis result of NS_B2_A1_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7855690, upper bound: 27.7839939
time: 0.64 seconds

## Relational analysis of NS_B2_A1_A1_B2_A2_B2_B2

### Relational analysis result of NS_B2_A1_A1_B2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7607951, upper bound: 27.7607880
time: 1.14 seconds

## BFS NS instance: NS_B2_A2_A1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -5.4029026, 18.6289616, -5.4029026, 18.6289616, -24.0318642, 24.0318642
1: -8.9004507, 18.9769077, -8.9004507, 18.9769077, -27.8773575, 27.8773575
2: -7.2595968, 20.5732021, -7.2595968, 20.5732021, -27.8327980, 27.8327980
3: -7.9015369, 28.2320576, -7.9015369, 28.2320576, -36.1335945, 36.1335945
4: -6.4061265, 26.6033630, -6.4061265, 26.6033630, -33.0094910, 33.0094910

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B2_A2_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_B2_A2_A1_A1_B1_B1_A1

### Relational analysis result of NS_B2_A2_A1_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.6956369, upper bound: 27.6966717
time: 0.58 seconds

## Relational analysis of NS_B2_A2_A1_A1_B1_B1_A2

### Relational analysis result of NS_B2_A2_A1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -26.7800986, upper bound: 27.8136820
time: 0.80 seconds

## BFS NS instance: NS_B2_A2_A1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -5.4029026, 18.6289616, -7.8388243, 25.1275501, -30.5304527, 26.4677849
1: -8.9004507, 18.9769077, -12.6399717, 25.7362232, -34.6366730, 31.6168747
2: -7.2595968, 20.5732021, -10.4622936, 27.6880188, -34.9476166, 31.0354900
3: -7.9015369, 28.2320576, -11.1317329, 37.9511490, -45.8526840, 39.3637848
4: -6.4061265, 26.6033630, -9.2031116, 35.9791756, -42.3852997, 35.8064728

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_B2_A2_A1_A1_B1_B2_A1

### Relational analysis result of NS_B2_A2_A1_A1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.6956371, upper bound: 27.6966717
time: 0.72 seconds

## Relational analysis of NS_B2_A2_A1_A1_B1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B2_A2_A1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B2_A2_A1_A1_B1_B2_B1

### Relational analysis result of NS_B2_A2_A1_A1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.1396345, upper bound: 27.2294674
time: 1.22 seconds

## Relational analysis of NS_B2_A2_A1_A1_B1_B2_B2

### Relational analysis result of NS_B2_A2_A1_A1_B1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -26.8759559, upper bound: 26.8594240
time: 1.24 seconds

## BFS NS instance: NS_B2_A2_A1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -5.4029026, 18.6289616, -5.7951403, 19.3845100, -24.7874126, 24.4241028
1: -8.9004507, 18.9769077, -9.4860306, 19.7728939, -28.6733437, 28.4629364
2: -7.2595968, 20.5732021, -7.7768764, 21.4003086, -28.6599045, 28.3500786
3: -7.9015369, 28.2320576, -8.3971539, 29.3123341, -37.2138710, 36.6292000
4: -6.4061265, 26.6033630, -6.8590794, 27.6703186, -34.0764389, 33.4624405

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_B2_A2_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B2_A2_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B2_A2_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B2_A2_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B2_A2_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B2_A2_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B2_A2_A1_A1_B2_B1_A1

### Relational analysis result of NS_B2_A2_A1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7780342, upper bound: 27.7802116
time: 0.88 seconds

## Relational analysis of NS_B2_A2_A1_A1_B2_B1_A2

### Relational analysis result of NS_B2_A2_A1_A1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7334938, upper bound: 27.7325533
time: 0.49 seconds

## BFS NS instance: NS_B2_A2_A1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -5.4029026, 18.6289616, -7.6980271, 24.1949196, -29.5978203, 26.3269882
1: -8.9004507, 18.9769077, -12.3796749, 24.7970486, -33.6974983, 31.3565807
2: -7.2595968, 20.5732021, -10.2556095, 26.6462612, -33.9058533, 30.8288116
3: -7.9015369, 28.2320576, -10.8816719, 36.5377197, -44.4392548, 39.1137314
4: -6.4061265, 26.6033630, -8.9968672, 34.6487350, -41.0548630, 35.6002312

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B2_A2_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B2_A2_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B2_A2_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B2_A2_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B2_A2_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_B2_A2_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B2_A2_A1_A1_B2_B2_A1

### Relational analysis result of NS_B2_A2_A1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7780342, upper bound: 27.7802116
time: 0.51 seconds

## Relational analysis of NS_B2_A2_A1_A1_B2_B2_A2

### Relational analysis result of NS_B2_A2_A1_A1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7334938, upper bound: 27.7325533
time: 0.86 seconds

## BFS NS instance: NS_B2_A2_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -7.8388243, 25.1275501, -5.4029026, 18.6289616, -26.4677849, 30.5304527
1: -12.6399717, 25.7362232, -8.9004507, 18.9769077, -31.6168728, 34.6366730
2: -10.4622936, 27.6880188, -7.2595968, 20.5732021, -31.0354958, 34.9476166
3: -11.1317329, 37.9511490, -7.9015369, 28.2320576, -39.3637848, 45.8526840
4: -9.2031116, 35.9791756, -6.4061265, 26.6033630, -35.8064728, 42.3852997

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B2_A2_A1_A2_B1_B1_A1

### Relational analysis result of NS_B2_A2_A1_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7361444, upper bound: 27.7380604
time: 0.50 seconds

## Relational analysis of NS_B2_A2_A1_A2_B1_B1_A2

### Relational analysis result of NS_B2_A2_A1_A2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7024386, upper bound: 27.6959108
time: 0.57 seconds

## BFS NS instance: NS_B2_A2_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -7.8388243, 25.1275501, -5.7951403, 19.3845100, -27.2233315, 30.9226837
1: -12.6399717, 25.7362232, -9.4860306, 19.7728939, -32.4128609, 35.2222519
2: -10.4622936, 27.6880188, -7.7768764, 21.4003086, -31.8626022, 35.4648933
3: -11.1317329, 37.9511490, -8.3971539, 29.3123341, -40.4440689, 46.3482971
4: -9.2031116, 35.9791756, -6.8590794, 27.6703186, -36.8734283, 42.8382568

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_A2_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B2_A2_A1_A2_B1_B2_A1

### Relational analysis result of NS_B2_A2_A1_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7361444, upper bound: 27.7588900
time: 0.55 seconds

## Relational analysis of NS_B2_A2_A1_A2_B1_B2_A2

### Relational analysis result of NS_B2_A2_A1_A2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7024386, upper bound: 27.7166187
time: 0.45 seconds

## BFS NS instance: NS_B2_A2_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -7.8388243, 25.1275501, -8.0542059, 25.8405075, -33.6793327, 33.1817551
1: -12.6399717, 25.7362232, -12.9825001, 26.5053539, -39.1453209, 38.7187233
2: -10.4622936, 27.6880188, -10.7663403, 28.4462528, -38.9085426, 38.4543571
3: -11.1317329, 37.9511490, -11.4391289, 38.9969482, -50.1286736, 49.3902779
4: -9.2031116, 35.9791756, -9.4587097, 36.9367409, -46.1398544, 45.4378777

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_A2_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B2_A2_A1_A2_B2_B1_B1

### Relational analysis result of NS_B2_A2_A1_A2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.6769005, upper bound: 27.6763837
time: 0.79 seconds

## Relational analysis of NS_B2_A2_A1_A2_B2_B1_B2

### Relational analysis result of NS_B2_A2_A1_A2_B2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.6641378, upper bound: 27.6641378
time: 0.53 seconds

## BFS NS instance: NS_B2_A2_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -7.8388243, 25.1275501, -8.3255482, 26.3078823, -34.1467018, 33.4530983
1: -12.6399717, 25.7362232, -13.3967314, 26.9927139, -39.6326790, 39.1329536
2: -10.4622936, 27.6880188, -11.1231251, 28.9440136, -39.4063072, 38.8111420
3: -11.1317329, 37.9511490, -11.7893782, 39.7058640, -50.8375931, 49.7405281
4: -9.2031116, 35.9791756, -9.7717667, 37.6607437, -46.8638535, 45.7509346

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B2_A2_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B2_A2_A1_A2_B2_B2_A1

### Relational analysis result of NS_B2_A2_A1_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.6991189, upper bound: 27.7467538
time: 0.74 seconds

## Relational analysis of NS_B2_A2_A1_A2_B2_B2_A2

### Relational analysis result of NS_B2_A2_A1_A2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.6641378, upper bound: 27.7097371
time: 0.83 seconds

## BFS NS instance: NS_B2_A2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -5.7951403, 19.3845100, -5.3988667, 18.6157188, -24.4108582, 24.7833767
1: -9.4860306, 19.7728939, -8.8940010, 18.9632626, -28.4492912, 28.6668949
2: -7.7768764, 21.4003086, -7.2540541, 20.5587864, -28.3356628, 28.6543598
3: -8.3971539, 29.3123341, -7.8957767, 28.2121086, -36.6092529, 37.2081108
4: -6.8590794, 27.6703186, -6.4012489, 26.5842133, -33.4432907, 34.0715637

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_B2_A2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B2_A2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B2_A2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B2_A2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_B2_A2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B2_A2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B2_A2_A2_B1_A1_B1_B1

### Relational analysis result of NS_B2_A2_A2_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7199958, upper bound: 27.7281647
time: 0.57 seconds

## Relational analysis of NS_B2_A2_A2_B1_A1_B1_B2

### Relational analysis result of NS_B2_A2_A2_B1_A1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7085500, upper bound: 27.7169988
time: 1.00 seconds

## BFS NS instance: NS_B2_A2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -5.7951403, 19.3845100, -7.3764977, 23.6040859, -29.3992214, 26.7610054
1: -9.4860306, 19.7728939, -11.8916283, 24.1629257, -33.6489563, 31.6645184
2: -7.7768764, 21.4003086, -9.8255701, 26.0249004, -33.8017769, 31.2258797
3: -8.3971539, 29.3123341, -10.4660425, 35.6526985, -44.0498505, 39.7783775
4: -6.8590794, 27.6703186, -8.6377630, 33.7726440, -40.6317215, 36.3080826

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_B2_A2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B2_A2_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B2_A2_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B2_A2_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B2_A2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_B2_A2_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B2_A2_A2_B1_A1_B2_B1

### Relational analysis result of NS_B2_A2_A2_B1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7199958, upper bound: 27.7281647
time: 0.79 seconds

## Relational analysis of NS_B2_A2_A2_B1_A1_B2_B2

### Relational analysis result of NS_B2_A2_A2_B1_A1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7085505, upper bound: 27.7169988
time: 0.73 seconds

## BFS NS instance: NS_B2_A2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -7.6980271, 24.1949196, -5.3877835, 18.5755424, -26.2735691, 29.5827026
1: -12.3796749, 24.7970486, -8.8748665, 18.9198151, -31.2994862, 33.6719131
2: -10.2556095, 26.6462612, -7.2388735, 20.5155830, -30.7711926, 33.8851318
3: -10.8816719, 36.5377197, -7.8773723, 28.1504784, -39.0321503, 44.4150887
4: -8.9968672, 34.6487350, -6.3855491, 26.5216045, -35.5184708, 41.0342827

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_B2_A2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B2_A2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B2_A2_A2_B1_A2_B1_A1

### Relational analysis result of NS_B2_A2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.4887132, upper bound: 27.4358982
time: 0.65 seconds

## Relational analysis of NS_B2_A2_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B2_A2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B2_A2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_B2_A2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_A2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B2_A2_A2_B1_A2_B1_B1

### Relational analysis result of NS_B2_A2_A2_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7337829, upper bound: 27.7368882
time: 1.31 seconds

## Relational analysis of NS_B2_A2_A2_B1_A2_B1_B2

### Relational analysis result of NS_B2_A2_A2_B1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7093093, upper bound: 27.7174336
time: 0.69 seconds

## BFS NS instance: NS_B2_A2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -7.6980271, 24.1949196, -7.3764977, 23.6040859, -31.3021126, 31.5714092
1: -12.3796749, 24.7970486, -11.8916283, 24.1629257, -36.5425987, 36.6886749
2: -10.2556095, 26.6462612, -9.8255701, 26.0249004, -36.2805099, 36.4718285
3: -10.8816719, 36.5377197, -10.4660425, 35.6526985, -46.5343704, 47.0037613
4: -8.9968672, 34.6487350, -8.6377630, 33.7726440, -42.7695122, 43.2864990

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_B2_A2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B2_A2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_B2_A2_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B2_A2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B2_A2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B2_A2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B2_A2_A2_B1_A2_B2_B1

### Relational analysis result of NS_B2_A2_A2_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7337829, upper bound: 27.7368882
time: 0.70 seconds

## Relational analysis of NS_B2_A2_A2_B1_A2_B2_B2

### Relational analysis result of NS_B2_A2_A2_B1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.6641378, upper bound: 27.7174336
time: 0.58 seconds

## BFS NS instance: NS_B2_A2_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -5.7951403, 19.3845100, -5.7951403, 19.3845100, -25.1796455, 25.1796455
1: -9.4860306, 19.7728939, -9.4860306, 19.7728939, -29.2589245, 29.2589245
2: -7.7768764, 21.4003086, -7.7768764, 21.4003086, -29.1771851, 29.1771851
3: -8.3971539, 29.3123341, -8.3971539, 29.3123341, -37.7094879, 37.7094841
4: -6.8590794, 27.6703186, -6.8590794, 27.6703186, -34.5293961, 34.5293961

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B2_A2_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_B2_A2_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B2_A2_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B2_A2_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B2_A2_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B2_A2_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B2_A2_A2_B2_B1_A1_A1

### Relational analysis result of NS_B2_A2_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7806981, upper bound: 27.7818802
time: 0.82 seconds

## Relational analysis of NS_B2_A2_A2_B2_B1_A1_A2

### Relational analysis result of NS_B2_A2_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7024386, upper bound: 27.7703937
time: 0.75 seconds

## BFS NS instance: NS_B2_A2_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -7.6980271, 24.1949196, -5.7951403, 19.3845100, -27.0825367, 29.9900551
1: -12.3796749, 24.7970486, -9.4860306, 19.7728939, -32.1525650, 34.2830811
2: -10.2556095, 26.6462612, -7.7768764, 21.4003086, -31.6559181, 34.4231300
3: -10.8816719, 36.5377197, -8.3971539, 29.3123341, -40.1940079, 44.9348717
4: -8.9968672, 34.6487350, -6.8590794, 27.6703186, -36.6671867, 41.5078125

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B2_A2_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_B2_A2_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B2_A2_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B2_A2_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B2_A2_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B2_A2_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B2_A2_A2_B2_B1_A2_A1

### Relational analysis result of NS_B2_A2_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7361445, upper bound: 27.7827570
time: 0.71 seconds

## Relational analysis of NS_B2_A2_A2_B2_B1_A2_A2

### Relational analysis result of NS_B2_A2_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7704266, upper bound: 27.7703937
time: 1.21 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 4.08 seconds
NS_B1_B1_A1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 0, lower bound: -27.7706916, upper bound: 27.7709281
NS_B1_B1_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 0, lower bound: -27.7706916, upper bound: 27.7709281
NS_B1_B1_A1_B2_A2_A1_B1, status: Status.VERIFIED, split count: 7, time: 4.08
Output dim: 0, lower bound: -27.7295923, upper bound: 27.7202259
NS_B1_B1_A1_B2_A2_A1_B2, status: Status.VERIFIED, split count: 7, time: 4.08
Output dim: 0, lower bound: -27.7295923, upper bound: 27.7202259
NS_B1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 0, lower bound: -27.7799429, upper bound: 27.7876283
NS_B1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 0, lower bound: -27.7799429, upper bound: 27.7876283
NS_B1_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.08
Output dim: 0, lower bound: -27.7682409, upper bound: 27.7685098
NS_B1_B1_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.08
Output dim: 0, lower bound: -27.7682409, upper bound: 27.7685098
NS_B1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 0, lower bound: -27.7706916, upper bound: 27.7844649
NS_B1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 0, lower bound: -27.7868138, upper bound: 27.7844649
NS_B1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 0, lower bound: -27.7706916, upper bound: 27.7844649
NS_B1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 0, lower bound: -27.7868138, upper bound: 27.7844649
NS_B1_B1_A2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 0, lower bound: -27.7751234, upper bound: 27.7739506
NS_B1_B1_A2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 0, lower bound: -27.7751234, upper bound: 27.7739506
NS_B1_B1_A2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 0, lower bound: -27.7222874, upper bound: 27.7739506
NS_B1_B1_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 0, lower bound: -27.7751234, upper bound: 27.7739506
NS_B2_A1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.08
Output dim: 0, lower bound: -27.7468375, upper bound: 27.7509987
NS_B2_A1_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.08
Output dim: 0, lower bound: -27.7607956, upper bound: 27.7607880
NS_B2_A1_A1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 0, lower bound: -27.7855690, upper bound: 27.7839939
NS_B2_A1_A1_B2_A2_B2_B2, status: Status.VERIFIED, split count: 7, time: 4.08
Output dim: 0, lower bound: -27.7607951, upper bound: 27.7607880
NS_B2_A2_A1_A1_B1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.08
Output dim: 0, lower bound: -27.6956369, upper bound: 27.6966717
NS_B2_A2_A1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 0, lower bound: -26.7800986, upper bound: 27.8136820
NS_B2_A2_A1_A1_B1_B2_B1, status: Status.VERIFIED, split count: 7, time: 4.08
Output dim: 0, lower bound: -27.1396345, upper bound: 27.2294674
NS_B2_A2_A1_A1_B1_B2_B2, status: Status.VERIFIED, split count: 7, time: 4.08
Output dim: 0, lower bound: -26.8759559, upper bound: 26.8594240
NS_B2_A2_A1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 0, lower bound: -27.7780342, upper bound: 27.7802116
NS_B2_A2_A1_A1_B2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.08
Output dim: 0, lower bound: -27.7334938, upper bound: 27.7325533
NS_B2_A2_A1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 0, lower bound: -27.7780342, upper bound: 27.7802116
NS_B2_A2_A1_A1_B2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.08
Output dim: 0, lower bound: -27.7334938, upper bound: 27.7325533
NS_B2_A2_A1_A2_B1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.08
Output dim: 0, lower bound: -27.7361444, upper bound: 27.7380604
NS_B2_A2_A1_A2_B1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.08
Output dim: 0, lower bound: -27.7024386, upper bound: 27.6959108
NS_B2_A2_A1_A2_B1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.08
Output dim: 0, lower bound: -27.7361444, upper bound: 27.7588900
NS_B2_A2_A1_A2_B1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.08
Output dim: 0, lower bound: -27.7024386, upper bound: 27.7166187
NS_B2_A2_A1_A2_B2_B1_B1, status: Status.VERIFIED, split count: 7, time: 4.08
Output dim: 0, lower bound: -27.6769005, upper bound: 27.6763837
NS_B2_A2_A1_A2_B2_B1_B2, status: Status.VERIFIED, split count: 7, time: 4.08
Output dim: 0, lower bound: -27.6641378, upper bound: 27.6641378
NS_B2_A2_A1_A2_B2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.08
Output dim: 0, lower bound: -27.6991189, upper bound: 27.7467538
NS_B2_A2_A1_A2_B2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.08
Output dim: 0, lower bound: -27.6641378, upper bound: 27.7097371
NS_B2_A2_A2_B1_A1_B1_B1, status: Status.VERIFIED, split count: 7, time: 4.08
Output dim: 0, lower bound: -27.7199958, upper bound: 27.7281647
NS_B2_A2_A2_B1_A1_B1_B2, status: Status.VERIFIED, split count: 7, time: 4.08
Output dim: 0, lower bound: -27.7085500, upper bound: 27.7169988
NS_B2_A2_A2_B1_A1_B2_B1, status: Status.VERIFIED, split count: 7, time: 4.08
Output dim: 0, lower bound: -27.7199958, upper bound: 27.7281647
NS_B2_A2_A2_B1_A1_B2_B2, status: Status.VERIFIED, split count: 7, time: 4.08
Output dim: 0, lower bound: -27.7085505, upper bound: 27.7169988
NS_B2_A2_A2_B1_A2_B1_B1, status: Status.VERIFIED, split count: 7, time: 4.08
Output dim: 0, lower bound: -27.7337829, upper bound: 27.7368882
NS_B2_A2_A2_B1_A2_B1_B2, status: Status.VERIFIED, split count: 7, time: 4.08
Output dim: 0, lower bound: -27.7093093, upper bound: 27.7174336
NS_B2_A2_A2_B1_A2_B2_B1, status: Status.VERIFIED, split count: 7, time: 4.08
Output dim: 0, lower bound: -27.7337829, upper bound: 27.7368882
NS_B2_A2_A2_B1_A2_B2_B2, status: Status.VERIFIED, split count: 7, time: 4.08
Output dim: 0, lower bound: -27.6641378, upper bound: 27.7174336
NS_B2_A2_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 0, lower bound: -27.7806981, upper bound: 27.7818802
NS_B2_A2_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 0, lower bound: -27.7024386, upper bound: 27.7703937
NS_B2_A2_A2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 0, lower bound: -27.7361445, upper bound: 27.7827570
NS_B2_A2_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.08
Output dim: 0, lower bound: -27.7704266, upper bound: 27.7703937

## BFS NS instance: NS_B1_B1_A1_B2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -5.0616093, 17.5872383, -4.6568985, 16.1854362, -21.2470455, 22.2441349
1: -8.3727589, 17.9134541, -7.7200298, 16.4539776, -24.8267326, 25.6334839
2: -6.8010635, 19.4322319, -6.2426319, 17.8746071, -24.6756687, 25.6748638
3: -7.4384723, 26.7205925, -6.8696055, 24.5522022, -31.9906693, 33.5901985
4: -5.9991608, 25.1323872, -5.4995651, 23.0713387, -29.0704994, 30.6319504

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 17

## BFS NS instance: NS_B1_B1_A1_B2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -5.0616093, 17.5872383, -4.6114683, 15.9813213, -21.0429306, 22.1987076
1: -8.3727589, 17.9134541, -7.6433921, 16.2519264, -24.6246815, 25.5568447
2: -6.8010635, 19.4322319, -6.1802125, 17.6584473, -24.4595070, 25.6124439
3: -7.4384723, 26.7205925, -6.8026519, 24.2351742, -31.6736393, 33.5232430
4: -5.9991608, 25.1323872, -5.4443140, 22.7835884, -28.7827492, 30.5767021

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 17

## BFS NS instance: NS_B1_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -5.6110678, 18.7911396, -4.4252939, 15.6111059, -21.2221737, 23.2164307
1: -9.1967754, 19.1523590, -7.3541880, 15.8749866, -25.0717621, 26.5065460
2: -7.5243778, 20.7475548, -5.9332600, 17.2657795, -24.7901573, 26.6808147
3: -8.1372128, 28.4492989, -6.5480304, 23.7588024, -31.8960152, 34.9973183
4: -6.6224542, 26.8305302, -5.2362361, 22.2943745, -28.9168224, 32.0667572

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 9

## BFS NS instance: NS_B1_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -5.5384359, 18.4686890, -4.4252939, 15.6111059, -21.1495419, 22.8939800
1: -9.0756388, 18.8374405, -7.3541880, 15.8749866, -24.9506207, 26.1916275
2: -7.4260931, 20.4082985, -5.9332600, 17.2657795, -24.6918716, 26.3415585
3: -8.0321388, 27.9505692, -6.5480304, 23.7588024, -31.7909412, 34.4986000
4: -6.5376554, 26.3716908, -5.2362361, 22.2943745, -28.8320274, 31.6079216

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 9

## BFS NS instance: NS_B1_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -5.6110678, 18.7911396, -4.6568985, 16.1854362, -21.7965050, 23.4480343
1: -9.1967754, 19.1523590, -7.7200298, 16.4539776, -25.6507530, 26.8723888
2: -7.5243778, 20.7475548, -6.2426319, 17.8746071, -25.3989849, 26.9901867
3: -8.1372128, 28.4492989, -6.8696055, 24.5522022, -32.6894150, 35.3189049
4: -6.6224542, 26.8305302, -5.4995651, 23.0713387, -29.6937923, 32.3300896

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B1_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_B1_A2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## BFS NS instance: NS_B1_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -5.5384359, 18.4686890, -4.6568985, 16.1854362, -21.7238731, 23.1255817
1: -9.0756388, 18.8374405, -7.7200298, 16.4539776, -25.5296135, 26.5574703
2: -7.4260931, 20.4082985, -6.2426319, 17.8746071, -25.3007011, 26.6509304
3: -8.0321388, 27.9505692, -6.8696055, 24.5522022, -32.5843430, 34.8201752
4: -6.5376554, 26.3716908, -5.4995651, 23.0713387, -29.6089935, 31.8712540

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B1_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_B1_A2_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## BFS NS instance: NS_B1_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -5.6110678, 18.7911396, -4.6114683, 15.9813213, -21.5923882, 23.4026070
1: -9.1967754, 19.1523590, -7.6433921, 16.2519264, -25.4487019, 26.7957516
2: -7.5243778, 20.7475548, -6.1802125, 17.6584473, -25.1828251, 26.9277649
3: -8.1372128, 28.4492989, -6.8026519, 24.2351742, -32.3723869, 35.2519455
4: -6.6224542, 26.8305302, -5.4443140, 22.7835884, -29.4060402, 32.2748451

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B1_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_B1_A2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## BFS NS instance: NS_B1_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -5.5384359, 18.4686890, -4.6114683, 15.9813213, -21.5197563, 23.0801563
1: -9.0756388, 18.8374405, -7.6433921, 16.2519264, -25.3275642, 26.4808311
2: -7.4260931, 20.4082985, -6.1802125, 17.6584473, -25.0845394, 26.5885086
3: -8.0321388, 27.9505692, -6.8026519, 24.2351742, -32.2673111, 34.7532196
4: -6.5376554, 26.3716908, -5.4443140, 22.7835884, -29.3212433, 31.8160057

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B1_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_B1_A2_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## BFS NS instance: NS_B1_B1_A2_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -7.5200233, 23.5865841, -4.6568985, 16.1854362, -23.7054596, 28.2434807
1: -12.0903320, 24.1655769, -7.7200298, 16.4539776, -28.5443096, 31.8856030
2: -10.0117388, 25.9833679, -6.2426319, 17.8746071, -27.8863449, 32.2259979
3: -10.6170120, 35.6386681, -6.8696055, 24.5522022, -35.1692123, 42.5082741
4: -8.7812548, 33.7779312, -5.4995651, 23.0713387, -31.8525810, 39.2774925

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 17

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B1_B1_A2_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_B1_A2_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 17

## BFS NS instance: NS_B1_B1_A2_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -7.5200233, 23.5865841, -4.6114683, 15.9813213, -23.5013447, 28.1980515
1: -12.0903320, 24.1655769, -7.6433921, 16.2519264, -28.3422585, 31.8089638
2: -10.0117388, 25.9833679, -6.1802125, 17.6584473, -27.6701851, 32.1635818
3: -10.6170120, 35.6386681, -6.8026519, 24.2351742, -34.8521843, 42.4413185
4: -8.7812548, 33.7779312, -5.4443140, 22.7835884, -31.5648308, 39.2222443

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 17

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B1_B1_A2_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_B1_A2_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 17

## BFS NS instance: NS_B1_B1_A2_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -7.4972787, 23.4641647, -4.6568985, 16.1854362, -23.6827145, 28.1210594
1: -12.0495205, 24.0436764, -7.7200298, 16.4539776, -28.5034962, 31.7637062
2: -9.9798918, 25.8561954, -6.2426319, 17.8746071, -27.8544998, 32.0988274
3: -10.5811272, 35.4473038, -6.8696055, 24.5522022, -35.1333313, 42.3169098
4: -8.7525120, 33.5995064, -5.4995651, 23.0713387, -31.8238449, 39.0990677

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 17

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_B1_A2_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 17

## BFS NS instance: NS_B1_B1_A2_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -7.4972787, 23.4641647, -4.6114683, 15.9813213, -23.4785995, 28.0756321
1: -12.0495205, 24.0436764, -7.6433921, 16.2519264, -28.3014412, 31.6870689
2: -9.9798918, 25.8561954, -6.1802125, 17.6584473, -27.6383400, 32.0364075
3: -10.5811272, 35.4473038, -6.8026519, 24.2351742, -34.8162994, 42.2499466
4: -8.7525120, 33.5995064, -5.4443140, 22.7835884, -31.5360947, 39.0438194

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 17

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_B1_A2_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 17

## BFS NS instance: NS_B2_A1_A1_B2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -4.7342229, 16.4243813, -7.5825920, 23.8449268, -28.5791492, 24.0069714
1: -7.8439040, 16.6983929, -12.1957550, 24.4371414, -32.2810440, 28.8941460
2: -6.3474259, 18.1358376, -10.1001081, 26.2645721, -32.6119995, 28.2359447
3: -6.9785309, 24.9128876, -10.7203646, 36.0134735, -42.9920006, 35.6332512
4: -5.5898933, 23.4206448, -8.8618498, 34.1424675, -39.7323608, 32.2824936

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B2_A1_A1_B2_A2_B2_B1_A1

### Relational analysis result of NS_B2_A1_A1_B2_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7141902, upper bound: 27.7509985
time: 0.78 seconds

## Relational analysis of NS_B2_A1_A1_B2_A2_B2_B1_A2

### Relational analysis result of NS_B2_A1_A1_B2_A2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7202259, upper bound: 27.7607880
time: 0.62 seconds

## BFS NS instance: NS_B2_A2_A1_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -5.2254572, 18.0865288, -5.4029026, 18.6289616, -23.8544197, 23.4894314
1: -8.6175680, 18.4240227, -8.9004507, 18.9769077, -27.5944729, 27.3244705
2: -7.0218673, 19.9814758, -7.2595968, 20.5732021, -27.5950699, 27.2410736
3: -7.6518474, 27.4162827, -7.9015369, 28.2320576, -35.8839035, 35.3178177
4: -6.1995420, 25.8183708, -6.4061265, 26.6033630, -32.8029022, 32.2244949

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_A2_A1_A1_B1_B1_A2_B1

### Relational analysis result of NS_B2_A2_A1_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8346065, upper bound: 27.8356115
time: 0.46 seconds

## Relational analysis of NS_B2_A2_A1_A1_B1_B1_A2_B2

### Relational analysis result of NS_B2_A2_A1_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8345355, upper bound: 27.8345355
time: 0.61 seconds

## BFS NS instance: NS_B2_A2_A1_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -5.3129044, 18.3465195, -5.7951403, 19.3845100, -24.6974144, 24.1416550
1: -8.7560854, 18.6885529, -9.4860306, 19.7728939, -28.5289803, 28.1745815
2: -7.1389112, 20.2666302, -7.7768764, 21.4003086, -28.5392189, 28.0435066
3: -7.7746172, 27.8083916, -8.3971539, 29.3123341, -37.0869522, 36.2055435
4: -6.3014946, 26.1958332, -6.8590794, 27.6703186, -33.9718132, 33.0549126

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B2_A2_A1_A1_B2_B1_A1_B1

### Relational analysis result of NS_B2_A2_A1_A1_B2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7603026, upper bound: 27.7614159
time: 0.78 seconds

## Relational analysis of NS_B2_A2_A1_A1_B2_B1_A1_B2

### Relational analysis result of NS_B2_A2_A1_A1_B2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7603026, upper bound: 27.7614159
time: 0.60 seconds

## BFS NS instance: NS_B2_A2_A1_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -5.3129044, 18.3465195, -7.6980271, 24.1949196, -29.5078239, 26.0445461
1: -8.7560854, 18.6885529, -12.3796749, 24.7970486, -33.5531349, 31.0682163
2: -7.1389112, 20.2666302, -10.2556095, 26.6462612, -33.7851715, 30.5222397
3: -7.7746172, 27.8083916, -10.8816719, 36.5377197, -44.3123360, 38.6900635
4: -6.3014946, 26.1958332, -8.9968672, 34.6487350, -40.9502296, 35.1926994

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B2_A2_A1_A1_B2_B2_A1_B1

### Relational analysis result of NS_B2_A2_A1_A1_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7284632, upper bound: 27.7264919
time: 0.65 seconds

## Relational analysis of NS_B2_A2_A1_A1_B2_B2_A1_B2

### Relational analysis result of NS_B2_A2_A1_A1_B2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7284632, upper bound: 27.7325533
time: 0.62 seconds

## BFS NS instance: NS_B2_A2_A2_B2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -5.7050543, 19.1141205, -5.7951403, 19.3845100, -25.0895653, 24.9092560
1: -9.3431530, 19.4964027, -9.4860306, 19.7728939, -29.1160469, 28.9824333
2: -7.6575027, 21.1045666, -7.7768764, 21.4003086, -29.0578098, 28.8814430
3: -8.2721643, 28.9063854, -8.3971539, 29.3123341, -37.5844994, 37.3035278
4: -6.7553849, 27.2787571, -6.8590794, 27.6703186, -34.4256897, 34.1378365

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B2_A2_A2_B2_B1_A1_A1_B1

### Relational analysis result of NS_B2_A2_A2_B2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7773083, upper bound: 27.7773083
time: 0.58 seconds

## Relational analysis of NS_B2_A2_A2_B2_B1_A1_A1_B2

### Relational analysis result of NS_B2_A2_A2_B2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7773083, upper bound: 27.7773083
time: 0.64 seconds

## BFS NS instance: NS_B2_A2_A2_B2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -5.6406951, 18.8145924, -5.7362509, 19.1925735, -24.8332691, 24.5508423
1: -9.2340069, 19.2059059, -9.3915825, 19.5786381, -28.8126450, 28.5974884
2: -7.5728989, 20.7903633, -7.6987038, 21.1923866, -28.7652855, 28.4890671
3: -8.1777573, 28.4431610, -8.3143616, 29.0225201, -37.2002716, 36.7575226
4: -6.6805844, 26.8537216, -6.7908206, 27.3934784, -34.0740623, 33.6445351

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B2_A2_A2_B2_B1_A1_A2_B1

### Relational analysis result of NS_B2_A2_A2_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7773083, upper bound: 27.7773083
time: 0.55 seconds

## Relational analysis of NS_B2_A2_A2_B2_B1_A1_A2_B2

### Relational analysis result of NS_B2_A2_A2_B2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7773083, upper bound: 27.7773083
time: 0.70 seconds

## BFS NS instance: NS_B2_A2_A2_B2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -7.5825920, 23.8449268, -5.7951403, 19.3845100, -26.9671021, 29.6400604
1: -12.1957550, 24.4371414, -9.4860306, 19.7728939, -31.9686470, 33.9231720
2: -10.1001081, 26.2645721, -7.7768764, 21.4003086, -31.5004158, 34.0414467
3: -10.7203646, 36.0134735, -8.3971539, 29.3123341, -40.0326996, 44.4106216
4: -8.8618498, 34.1424675, -6.8590794, 27.6703186, -36.5321617, 41.0015488

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B2_A2_A2_B2_B1_A2_A1_B1

### Relational analysis result of NS_B2_A2_A2_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7704266, upper bound: 27.7703937
time: 1.03 seconds

## Relational analysis of NS_B2_A2_A2_B2_B1_A2_A1_B2

### Relational analysis result of NS_B2_A2_A2_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7704266, upper bound: 27.7703937
time: 0.70 seconds

## BFS NS instance: NS_B2_A2_A2_B2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -7.4629340, 23.4205666, -5.7362509, 19.1925735, -26.6555080, 29.1568184
1: -12.0006990, 24.0046806, -9.3915825, 19.5786381, -31.5793381, 33.3962631
2: -9.9388428, 25.8092690, -7.6987038, 21.1923866, -31.1312294, 33.5079727
3: -10.5490026, 35.3722382, -8.3143616, 29.0225201, -39.5715218, 43.6865959
4: -8.7199173, 33.5306396, -6.7908206, 27.3934784, -36.1133957, 40.3214569

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B2_A2_A2_B2_B1_A2_A2_B1

### Relational analysis result of NS_B2_A2_A2_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7704266, upper bound: 27.7703937
time: 0.85 seconds

## Relational analysis of NS_B2_A2_A2_B2_B1_A2_A2_B2

### Relational analysis result of NS_B2_A2_A2_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7704266, upper bound: 27.7703937
time: 0.83 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 2.50 seconds
NS_B2_A1_A1_B2_A2_B2_B1_A1, status: Status.VERIFIED, split count: 8, time: 2.50
Output dim: 0, lower bound: -27.7141902, upper bound: 27.7509985
NS_B2_A1_A1_B2_A2_B2_B1_A2, status: Status.VERIFIED, split count: 8, time: 2.50
Output dim: 0, lower bound: -27.7202259, upper bound: 27.7607880
NS_B2_A2_A1_A1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -27.8346065, upper bound: 27.8356115
NS_B2_A2_A1_A1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -27.8345355, upper bound: 27.8345355
NS_B2_A2_A1_A1_B2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 2.50
Output dim: 0, lower bound: -27.7603026, upper bound: 27.7614159
NS_B2_A2_A1_A1_B2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 2.50
Output dim: 0, lower bound: -27.7603026, upper bound: 27.7614159
NS_B2_A2_A1_A1_B2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 2.50
Output dim: 0, lower bound: -27.7284632, upper bound: 27.7264919
NS_B2_A2_A1_A1_B2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 2.50
Output dim: 0, lower bound: -27.7284632, upper bound: 27.7325533
NS_B2_A2_A2_B2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -27.7773083, upper bound: 27.7773083
NS_B2_A2_A2_B2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -27.7773083, upper bound: 27.7773083
NS_B2_A2_A2_B2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -27.7773083, upper bound: 27.7773083
NS_B2_A2_A2_B2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -27.7773083, upper bound: 27.7773083
NS_B2_A2_A2_B2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -27.7704266, upper bound: 27.7703937
NS_B2_A2_A2_B2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -27.7704266, upper bound: 27.7703937
NS_B2_A2_A2_B2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -27.7704266, upper bound: 27.7703937
NS_B2_A2_A2_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -27.7704266, upper bound: 27.7703937

## BFS NS instance: NS_B2_A2_A1_A1_B1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -4.5859890, 16.1962795, -3.7183187, 13.7603569, -18.3463421, 19.9145966
1: -7.6056671, 16.4772587, -6.2369590, 13.9218893, -21.5275574, 22.7142162
2: -6.1667538, 17.9085217, -4.9797230, 15.1893826, -21.3561306, 22.8882446
3: -6.7579288, 24.6004581, -5.5407190, 20.9493256, -27.7072544, 30.1411781
4: -5.4389095, 23.0918312, -4.3884597, 19.4684181, -24.9073257, 27.4802914

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_A2_A1_A1_B1_B1_A2_B1_A1

### Relational analysis result of NS_B2_A2_A1_A1_B1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8345355, upper bound: 27.8345355
time: 0.57 seconds

## Relational analysis of NS_B2_A2_A1_A1_B1_B1_A2_B1_A2

### Relational analysis result of NS_B2_A2_A1_A1_B1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8345355, upper bound: 27.8345355
time: 1.12 seconds

## BFS NS instance: NS_B2_A2_A1_A1_B1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -5.2147598, 18.0550041, -4.8010516, 16.8056908, -22.0204468, 22.8560562
1: -8.6007614, 18.3914089, -7.9506578, 17.0972404, -25.6980019, 26.3420639
2: -7.0076284, 19.9470100, -6.4578629, 18.5809574, -25.5885811, 26.4048710
3: -7.6369715, 27.3689270, -7.0589008, 25.4944801, -33.1314507, 34.4278259
4: -6.1869955, 25.7728291, -5.7009921, 23.9738159, -30.1608124, 31.4738121

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_A2_A1_A1_B1_B1_A2_B2_A1

### Relational analysis result of NS_B2_A2_A1_A1_B1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8345355, upper bound: 27.8345355
time: 0.83 seconds

## Relational analysis of NS_B2_A2_A1_A1_B1_B1_A2_B2_A2

### Relational analysis result of NS_B2_A2_A1_A1_B1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8345355, upper bound: 27.8345355
time: 0.81 seconds

## BFS NS instance: NS_B2_A2_A2_B2_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -5.7050543, 19.1141205, -5.7050543, 19.1141205, -24.8191757, 24.8191757
1: -9.3431530, 19.4964027, -9.3431530, 19.4964027, -28.8395538, 28.8395538
2: -7.6575027, 21.1045666, -7.6575027, 21.1045666, -28.7620697, 28.7620697
3: -8.2721643, 28.9063854, -8.2721643, 28.9063854, -37.1785507, 37.1785507
4: -6.7553849, 27.2787571, -6.7553849, 27.2787571, -34.0341377, 34.0341339

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B2_A2_A2_B2_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## BFS NS instance: NS_B2_A2_A2_B2_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -5.7050543, 19.1141205, -5.6406951, 18.8145924, -24.5196457, 24.7548161
1: -9.3431530, 19.4964027, -9.2340069, 19.2059059, -28.5490570, 28.7304096
2: -7.6575027, 21.1045666, -7.5728989, 20.7903633, -28.4478664, 28.6774654
3: -8.2721643, 28.9063854, -8.1777573, 28.4431610, -36.7153244, 37.0841370
4: -6.7553849, 27.2787571, -6.6805844, 26.8537216, -33.6090927, 33.9593430

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B2_A2_A2_B2_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## BFS NS instance: NS_B2_A2_A2_B2_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -5.6406951, 18.8145924, -5.7050543, 19.1141205, -24.7548161, 24.5196457
1: -9.2340069, 19.2059059, -9.3431530, 19.4964027, -28.7304096, 28.5490570
2: -7.5728989, 20.7903633, -7.6575027, 21.1045666, -28.6774654, 28.4478664
3: -8.1777573, 28.4431610, -8.2721643, 28.9063854, -37.0841370, 36.7153244
4: -6.6805844, 26.8537216, -6.7553849, 27.2787571, -33.9593391, 33.6090965

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B2_A2_A2_B2_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## BFS NS instance: NS_B2_A2_A2_B2_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -5.6406951, 18.8145924, -5.6406951, 18.8145924, -24.4552879, 24.4552879
1: -9.2340069, 19.2059059, -9.2340069, 19.2059059, -28.4399128, 28.4399128
2: -7.5728989, 20.7903633, -7.5728989, 20.7903633, -28.3632622, 28.3632622
3: -8.1777573, 28.4431610, -8.1777573, 28.4431610, -36.6209183, 36.6209183
4: -6.6805844, 26.8537216, -6.6805844, 26.8537216, -33.5342941, 33.5342941

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B2_A2_A2_B2_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## BFS NS instance: NS_B2_A2_A2_B2_B1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -7.5825920, 23.8449268, -5.7050543, 19.1141205, -26.6967125, 29.5499802
1: -12.1957550, 24.4371414, -9.3431530, 19.4964027, -31.6921539, 33.7802925
2: -10.1001081, 26.2645721, -7.6575027, 21.1045666, -31.2046738, 33.9220734
3: -10.7203646, 36.0134735, -8.2721643, 28.9063854, -39.6267509, 44.2856369
4: -8.8618498, 34.1424675, -6.7553849, 27.2787571, -36.1406059, 40.8978424

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B2_A2_A2_B2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## BFS NS instance: NS_B2_A2_A2_B2_B1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -7.5825920, 23.8449268, -5.6406951, 18.8145924, -26.3971844, 29.4856205
1: -12.1957550, 24.4371414, -9.2340069, 19.2059059, -31.4016571, 33.6711502
2: -10.1001081, 26.2645721, -7.5728989, 20.7903633, -30.8904724, 33.8374710
3: -10.7203646, 36.0134735, -8.1777573, 28.4431610, -39.1635246, 44.1912270
4: -8.8618498, 34.1424675, -6.6805844, 26.8537216, -35.7155609, 40.8230476

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B2_A2_A2_B2_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## BFS NS instance: NS_B2_A2_A2_B2_B1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -7.4629340, 23.4205666, -5.7050543, 19.1141205, -26.5770550, 29.1256218
1: -12.0006990, 24.0046806, -9.3431530, 19.4964027, -31.4971008, 33.3478317
2: -9.9388428, 25.8092690, -7.6575027, 21.1045666, -31.0434074, 33.4667702
3: -10.5490026, 35.3722382, -8.2721643, 28.9063854, -39.4553795, 43.6444016
4: -8.7199173, 33.5306396, -6.7553849, 27.2787571, -35.9986725, 40.2860146

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B2_A2_A2_B2_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## BFS NS instance: NS_B2_A2_A2_B2_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -7.4629340, 23.4205666, -5.6406951, 18.8145924, -26.2775269, 29.0612621
1: -12.0006990, 24.0046806, -9.2340069, 19.2059059, -31.2066040, 33.2386856
2: -9.9388428, 25.8092690, -7.5728989, 20.7903633, -30.7292061, 33.3821678
3: -10.5490026, 35.3722382, -8.1777573, 28.4431610, -38.9921646, 43.5499916
4: -8.7199173, 33.5306396, -6.6805844, 26.8537216, -35.5736389, 40.2112160

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B2_A2_A2_B2_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 2.34 + 220.05 = 222.39 seconds
