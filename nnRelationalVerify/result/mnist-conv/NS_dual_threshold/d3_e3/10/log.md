## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 10)
Time budget: 600 seconds
Split limit: 100
Threshold: 1.0349644734999999


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-5.3545036, -2.6462827, -5.3545036, -2.6462827, -2.2198710, 2.2198710)
1: (-6.3247952, -4.2507443, -6.3247952, -4.2507443, -1.7507243, 1.7507243)
2: (-4.6537833, -2.6294413, -4.6537833, -2.6294413, -1.5917411, 1.5917413)
3: (-7.8594894, -5.0927763, -7.8594894, -5.0927763, -2.0713673, 2.0713677)
4: (-11.8233414, -9.0321541, -11.8233414, -9.0321541, -2.3726444, 2.3726444)
5: (-6.3656282, -4.2320991, -6.3656282, -4.2320991, -1.7345924, 1.7345924)
6: (-10.4613619, -7.9367504, -10.4613619, -7.9367504, -2.0136123, 2.0136118)
7: (-2.8968585, -0.7577722, -2.8968585, -0.7577722, -1.8201418, 1.8201420)
8: (1.9637957, 3.6150055, 1.9637957, 3.6150055, -1.3549571, 1.3549573)
9: (-8.0759621, -5.5572004, -8.0759621, -5.5572004, -2.0553026, 2.0553021)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.68 + 33.75 = 56.42 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -1.0401645, upper bound: 1.0401642

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 5871
type: B, layer: 1, pos: 5871
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 4571
type: A, layer: 1, pos: 4571
type: A, layer: 1, pos: 5856
type: B, layer: 1, pos: 5856
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 4572
type: B, layer: 1, pos: 4572
type: B, layer: 1, pos: 4573
type: A, layer: 1, pos: 4573
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 5858
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 5801
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 6139
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 887
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 957

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0285205, upper bound: 1.0393477
time: 4.14 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0393471, upper bound: 1.0393484
time: 4.82 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 9.06 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 9.06
Output dim: 7, lower bound: -1.0285205, upper bound: 1.0393477
NS_A2, status: Status.UNKNOWN, split count: 1, time: 9.06
Output dim: 7, lower bound: -1.0393471, upper bound: 1.0393484

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -5.3448629, -2.6513064, -5.3535991, -2.6487818, -2.2063532, 2.2138548
1: -6.3199863, -4.2584815, -6.3242135, -4.2543473, -1.7423882, 1.7425404
2: -4.6420040, -2.6378279, -4.6528454, -2.6335866, -1.5757818, 1.5829291
3: -7.8586378, -5.0940123, -7.8584652, -5.0932484, -2.0700850, 2.0685639
4: -11.8088398, -9.0452976, -11.8159161, -9.0326414, -2.3578911, 2.3510776
5: -6.3616781, -4.2332859, -6.3642983, -4.2322807, -1.7303658, 1.7315521
6: -10.4562159, -7.9401374, -10.4607811, -7.9384832, -2.0048246, 2.0096078
7: -2.8744102, -0.7741427, -2.8859327, -0.7579317, -1.7970166, 1.7905664
8: 1.9756560, 3.6014438, 1.9642277, 3.6082225, -1.3354609, 1.3401518
9: -8.0701504, -5.5636220, -8.0728464, -5.5576634, -2.0488563, 2.0413256

Time for backsubstitution: 22.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5871
type: A, layer: 1, pos: 5871
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 4571
type: A, layer: 1, pos: 4571
type: A, layer: 1, pos: 5856
type: B, layer: 1, pos: 5856
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 4572
type: B, layer: 1, pos: 4572
type: A, layer: 1, pos: 4573
type: B, layer: 1, pos: 4573
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 5858
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 5801
type: A, layer: 1, pos: 5801
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 887
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 5871

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5871

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0270879, upper bound: 1.0354523
time: 4.25 seconds

## Relational analysis of NS_A1_A2

### Relational analysis result of NS_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0285150, upper bound: 1.0393427
time: 24.87 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -5.3544674, -2.6463146, -5.3544798, -2.6463041, -2.2198105, 2.2110400
1: -6.3247867, -4.2507839, -6.3247857, -4.2507639, -1.7506957, 1.7443419
2: -4.6537824, -2.6294842, -4.6537824, -2.6294675, -1.5917215, 1.5871401
3: -7.8594294, -5.0936785, -7.8594561, -5.0931931, -2.0707273, 2.0737429
4: -11.8232851, -9.0321836, -11.8232985, -9.0321684, -2.3500996, 2.3721294
5: -6.3656182, -4.2321053, -6.3656225, -4.2321062, -1.7331562, 1.7344773
6: -10.4613619, -7.9368286, -10.4613628, -7.9367905, -2.0128613, 2.0066915
7: -2.8968077, -0.7577724, -2.8968346, -0.7577713, -1.7883425, 1.8193042
8: 1.9638104, 3.6149974, 1.9638042, 3.6150012, -1.3549359, 1.3355761
9: -8.0758963, -5.5572109, -8.0759287, -5.5572100, -2.0409937, 2.0527639

Time for backsubstitution: 21.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5871
type: B, layer: 1, pos: 5871
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 4571
type: B, layer: 1, pos: 4571
type: A, layer: 1, pos: 5856
type: B, layer: 1, pos: 5856
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 4572
type: A, layer: 1, pos: 4572
type: B, layer: 1, pos: 4573
type: A, layer: 1, pos: 4573
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 5858
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 5801
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 887
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 5871

## Relational analysis of NS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5871

### Candidate
type: B, layer: 1, pos: 4626

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0354518, upper bound: 1.0379158
time: 4.46 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0393413, upper bound: 1.0393429
time: 4.51 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 36.30 seconds
NS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 36.30
Output dim: 7, lower bound: -1.0270879, upper bound: 1.0354523
NS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 36.30
Output dim: 7, lower bound: -1.0285150, upper bound: 1.0393427
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 36.30
Output dim: 7, lower bound: -1.0354518, upper bound: 1.0379158
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 36.30
Output dim: 7, lower bound: -1.0393413, upper bound: 1.0393429

## BFS NS instance: NS_A1_A1

### Backsubstitution after applying NS history:
0: -5.3147545, -2.6629608, -5.3399463, -2.6506157, -2.1734424, 2.1843343
1: -6.3093019, -4.2745342, -6.3215342, -4.2611132, -1.7186584, 1.7193747
2: -4.6238222, -2.6541886, -4.6481128, -2.6409750, -1.5459733, 1.5592322
3: -7.8350868, -5.1136618, -7.8482046, -5.0946670, -2.0424643, 2.0358620
4: -11.7832813, -9.0724201, -11.8038311, -9.0376368, -2.3260221, 2.3111248
5: -6.3433881, -4.2540445, -6.3564663, -4.2350249, -1.7089229, 1.7017341
6: -10.4353619, -7.9554920, -10.4511280, -7.9414196, -1.9798856, 1.9819555
7: -2.8597291, -0.7879517, -2.8832624, -0.7645626, -1.7747741, 1.7732973
8: 1.9903951, 3.5947852, 1.9710178, 3.6078281, -1.3178399, 1.3237457
9: -8.0408239, -5.5954666, -8.0599499, -5.5633249, -2.0140924, 1.9961057

Time for backsubstitution: 21.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5871
type: B, layer: 1, pos: 5871
type: A, layer: 1, pos: 4571
type: B, layer: 1, pos: 4571
type: A, layer: 1, pos: 5856
type: B, layer: 1, pos: 5856
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 4572
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 4573
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 5858
type: A, layer: 1, pos: 5858
type: B, layer: 1, pos: 5801
type: A, layer: 1, pos: 5801
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 887
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 5871

### Candidate
type: B, layer: 1, pos: 5871

### Candidate
type: A, layer: 1, pos: 4571

## Relational analysis of NS_A1_A1_A1

### Relational analysis result of NS_A1_A1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0243630, upper bound: 1.0283968
time: 4.46 seconds

## Relational analysis of NS_A1_A1_A2

### Relational analysis result of NS_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0270768, upper bound: 1.0354406
time: 5.49 seconds

## BFS NS instance: NS_A1_A2

### Backsubstitution after applying NS history:
0: -5.3448343, -2.6513073, -5.3535862, -2.6487839, -2.2056384, 2.2170739
1: -6.3199806, -4.2584958, -6.3242111, -4.2543540, -1.7350101, 1.7488508
2: -4.6419959, -2.6378398, -4.6528416, -2.6335914, -1.5749102, 1.5771580
3: -7.8586154, -5.0940161, -7.8584547, -5.0932493, -2.0587621, 2.0685525
4: -11.8088093, -9.0453081, -11.8159018, -9.0326443, -2.3574085, 2.3510628
5: -6.3616600, -4.2332926, -6.3642921, -4.2322831, -1.7299476, 1.7315419
6: -10.4561968, -7.9401417, -10.4607716, -7.9384880, -2.0011158, 2.0095963
7: -2.8744054, -0.7741528, -2.8859308, -0.7579350, -1.7970071, 1.7837293
8: 1.9756722, 3.6014428, 1.9642334, 3.6082225, -1.3261433, 1.3385117
9: -8.0701141, -5.5636311, -8.0728312, -5.5576677, -2.0444613, 2.0413074

Time for backsubstitution: 21.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5871
type: A, layer: 1, pos: 5871
type: B, layer: 1, pos: 4571
type: A, layer: 1, pos: 4571
type: B, layer: 1, pos: 5856
type: A, layer: 1, pos: 5856
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 4572
type: B, layer: 1, pos: 4572
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 5858
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 5801
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 6139
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 5871

### Candidate
type: A, layer: 1, pos: 5871

### Candidate
type: B, layer: 1, pos: 4571

## Relational analysis of NS_A1_A2_B1

### Relational analysis result of NS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0214377, upper bound: 1.0366154
time: 4.16 seconds

## Relational analysis of NS_A1_A2_B2

### Relational analysis result of NS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0285032, upper bound: 1.0393307
time: 4.19 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -5.3408117, -2.6481497, -5.3243508, -2.6579530, -2.1902790, 2.1781092
1: -6.3221045, -4.2575498, -6.3140917, -4.2668042, -1.7275476, 1.7206397
2: -4.6490269, -2.6368704, -4.6355739, -2.6458182, -1.5680113, 1.5573239
3: -7.8491707, -5.0950971, -7.8359308, -5.1128483, -2.0380092, 2.0461421
4: -11.8112059, -9.0371819, -11.7977591, -9.0593147, -2.3101377, 2.3402548
5: -6.3577843, -4.2348466, -6.3473759, -4.2528663, -1.7032242, 1.7130842
6: -10.4517126, -7.9397616, -10.4405069, -7.9521484, -1.9851923, 1.9817557
7: -2.8941326, -0.7644036, -2.8821206, -0.7715831, -1.7710700, 1.7970400
8: 1.9706044, 3.6146030, 1.9785414, 3.6083436, -1.3385262, 1.3179579
9: -8.0630016, -5.5628786, -8.0466022, -5.5890565, -1.9957733, 2.0180297

Time for backsubstitution: 21.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5871
type: B, layer: 1, pos: 5871
type: A, layer: 1, pos: 4571
type: B, layer: 1, pos: 4571
type: B, layer: 1, pos: 5856
type: A, layer: 1, pos: 5856
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 4572
type: B, layer: 1, pos: 4572
type: B, layer: 1, pos: 4573
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 5858
type: B, layer: 1, pos: 5858
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 5801
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 887
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 5871

### Candidate
type: B, layer: 1, pos: 5871

### Candidate
type: A, layer: 1, pos: 4571

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0327278, upper bound: 1.0308597
time: 4.36 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0354407, upper bound: 1.0379039
time: 4.50 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -5.3544545, -2.6463153, -5.3544497, -2.6463056, -2.2230315, 2.2103238
1: -6.3247828, -4.2507896, -6.3247800, -4.2507782, -1.7570109, 1.7369685
2: -4.6537781, -2.6294892, -4.6537752, -2.6294787, -1.5859065, 1.5862708
3: -7.8594198, -5.0936799, -7.8594332, -5.0931959, -2.0707150, 2.0624194
4: -11.8232727, -9.0321856, -11.8232708, -9.0321741, -2.3500824, 2.3716459
5: -6.3656101, -4.2321076, -6.3656054, -4.2321091, -1.7331467, 1.7340605
6: -10.4613543, -7.9368329, -10.4613447, -7.9367957, -2.0128484, 2.0029826
7: -2.8968058, -0.7577770, -2.8968291, -0.7577820, -1.7815051, 1.8192933
8: 1.9638171, 3.6149974, 1.9638190, 3.6150002, -1.3532965, 1.3262591
9: -8.0758781, -5.5572138, -8.0758896, -5.5572195, -2.0409756, 2.0483699

Time for backsubstitution: 22.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5871
type: B, layer: 1, pos: 5871
type: A, layer: 1, pos: 4571
type: B, layer: 1, pos: 4571
type: A, layer: 1, pos: 5856
type: B, layer: 1, pos: 5856
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 4572
type: A, layer: 1, pos: 4572
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 5858
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 5801
type: B, layer: 1, pos: 5801
type: A, layer: 1, pos: 6139
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 887
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 5871

### Candidate
type: B, layer: 1, pos: 5871

### Candidate
type: A, layer: 1, pos: 4571

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0366149, upper bound: 1.0322657
time: 6.38 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0393302, upper bound: 1.0393310
time: 4.59 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 33.33 seconds
NS_A1_A1_A1, status: Status.VERIFIED, split count: 3, time: 33.33
Output dim: 7, lower bound: -1.0243630, upper bound: 1.0283968
NS_A1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 33.33
Output dim: 7, lower bound: -1.0270768, upper bound: 1.0354406
NS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 33.33
Output dim: 7, lower bound: -1.0214377, upper bound: 1.0366154
NS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 33.33
Output dim: 7, lower bound: -1.0285032, upper bound: 1.0393307
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 33.33
Output dim: 7, lower bound: -1.0327278, upper bound: 1.0308597
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 33.33
Output dim: 7, lower bound: -1.0354407, upper bound: 1.0379039
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 33.33
Output dim: 7, lower bound: -1.0366149, upper bound: 1.0322657
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 33.33
Output dim: 7, lower bound: -1.0393302, upper bound: 1.0393310

## BFS NS instance: NS_A1_A1_A2

### Backsubstitution after applying NS history:
0: -5.3147483, -2.6629734, -5.3399434, -2.6506214, -2.1717615, 2.1838069
1: -6.3092556, -4.2745395, -6.3215151, -4.2611170, -1.6989427, 1.7193480
2: -4.6238184, -2.6542044, -4.6481109, -2.6409826, -1.5457582, 1.5509777
3: -7.8350787, -5.1137114, -7.8482018, -5.0946894, -2.0424347, 2.0076447
4: -11.7832479, -9.0724287, -11.8038177, -9.0376434, -2.3114252, 2.3101072
5: -6.3433838, -4.2540593, -6.3564649, -4.2350316, -1.7089128, 1.6998127
6: -10.4352989, -7.9555006, -10.4510984, -7.9414220, -1.9356904, 1.9808440
7: -2.8597174, -0.7879734, -2.8832579, -0.7645710, -1.7741027, 1.7566257
8: 1.9903970, 3.5947585, 1.9710197, 3.6078167, -1.3175402, 1.2977178
9: -8.0408096, -5.5954714, -8.0599442, -5.5633278, -2.0173702, 1.9960856

Time for backsubstitution: 21.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5871
type: B, layer: 1, pos: 5871
type: A, layer: 1, pos: 5856
type: B, layer: 1, pos: 5856
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 4572
type: A, layer: 1, pos: 4572
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 5858
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 4571
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 5801
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 5858
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 887
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 6139

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 5871

### Candidate
type: B, layer: 1, pos: 5871

### Candidate
type: A, layer: 1, pos: 5856

## Relational analysis of NS_A1_A1_A2_A1

### Relational analysis result of NS_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0213511, upper bound: 1.0354063
time: 6.19 seconds

## Relational analysis of NS_A1_A1_A2_A2

### Relational analysis result of NS_A1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0270392, upper bound: 1.0354069
time: 4.24 seconds

## BFS NS instance: NS_A1_A2_B1

### Backsubstitution after applying NS history:
0: -5.3409338, -2.6554322, -5.3431516, -2.6578755, -2.1884127, 2.1962972
1: -6.3055277, -4.2615190, -6.2938404, -4.2743731, -1.7002144, 1.7153788
2: -4.6407127, -2.6447325, -4.6446996, -2.6486137, -1.5573602, 1.5615010
3: -7.8552313, -5.1094213, -7.8338914, -5.1246099, -2.0240788, 2.0288043
4: -11.7988167, -9.0497665, -11.7948036, -9.0505447, -2.3285885, 2.3250685
5: -6.3602676, -4.2378154, -6.3581071, -4.2421298, -1.7169695, 1.7192690
6: -10.4347506, -7.9434376, -10.4170818, -7.9678249, -1.9481740, 1.9614363
7: -2.8690786, -0.7834594, -2.8658628, -0.7767403, -1.7726054, 1.7530963
8: 1.9768786, 3.5901766, 1.9795923, 3.5855441, -1.3016918, 1.3112764
9: -8.0652323, -5.5664773, -8.0586929, -5.5638552, -2.0304847, 2.0237856

Time for backsubstitution: 22.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5871
type: A, layer: 1, pos: 5871
type: B, layer: 1, pos: 5856
type: A, layer: 1, pos: 5856
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 4571
type: A, layer: 1, pos: 4572
type: B, layer: 1, pos: 4572
type: B, layer: 1, pos: 4573
type: A, layer: 1, pos: 4573
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 5858
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 6139
type: B, layer: 1, pos: 5801
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 887
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 5871

### Candidate
type: A, layer: 1, pos: 5871

### Candidate
type: B, layer: 1, pos: 5856

## Relational analysis of NS_A1_A2_B1_B1

### Relational analysis result of NS_A1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0214015, upper bound: 1.0308820
time: 4.40 seconds

## Relational analysis of NS_A1_A2_B1_B2

### Relational analysis result of NS_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0214015, upper bound: 1.0365806
time: 4.32 seconds

## BFS NS instance: NS_A1_A2_B2

### Backsubstitution after applying NS history:
0: -5.3448315, -2.6513133, -5.3535781, -2.6487966, -2.2051096, 2.2154045
1: -6.3199625, -4.2584982, -6.3241649, -4.2543612, -1.7343767, 1.7291336
2: -4.6419950, -2.6378472, -4.6528387, -2.6336079, -1.5666618, 1.5769398
3: -7.8586116, -5.0940361, -7.8584471, -5.0932999, -2.0305448, 2.0643084
4: -11.8087978, -9.0453110, -11.8158693, -9.0326538, -2.3563786, 2.3364682
5: -6.3616595, -4.2332964, -6.3642883, -4.2322969, -1.7280273, 1.7315319
6: -10.4561644, -7.9401441, -10.4607086, -7.9384937, -2.0000067, 1.9653978
7: -2.8744001, -0.7741628, -2.8859191, -0.7579553, -1.7803435, 1.7830439
8: 1.9756742, 3.6014314, 1.9642382, 3.6081948, -1.3001153, 1.3366914
9: -8.0701084, -5.5636330, -8.0728168, -5.5576739, -2.0444412, 2.0445457

Time for backsubstitution: 22.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5871
type: A, layer: 1, pos: 5871
type: B, layer: 1, pos: 5856
type: A, layer: 1, pos: 5856
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 4572
type: B, layer: 1, pos: 4572
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 4573
type: A, layer: 1, pos: 5858
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 4571
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 5801
type: A, layer: 1, pos: 5801
type: B, layer: 1, pos: 5858
type: A, layer: 1, pos: 887
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 6139

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 5871

### Candidate
type: A, layer: 1, pos: 5871

### Candidate
type: B, layer: 1, pos: 5856

## Relational analysis of NS_A1_A2_B2_B1

### Relational analysis result of NS_A1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0284669, upper bound: 1.0335970
time: 4.44 seconds

## Relational analysis of NS_A1_A2_B2_B2

### Relational analysis result of NS_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0284669, upper bound: 1.0392959
time: 4.48 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -5.3408051, -2.6481616, -5.3243494, -2.6579576, -2.1885996, 2.1775794
1: -6.3220577, -4.2575564, -6.3140717, -4.2668061, -1.7078304, 1.7206116
2: -4.6490235, -2.6368876, -4.6355739, -2.6458249, -1.5676281, 1.5490746
3: -7.8491640, -5.0951476, -7.8359270, -5.1128702, -2.0379806, 2.0179238
4: -11.8111725, -9.0371904, -11.7977467, -9.0593185, -2.2955208, 2.3392229
5: -6.3577800, -4.2348599, -6.3473740, -4.2528720, -1.7032137, 1.7111640
6: -10.4516497, -7.9397693, -10.4404755, -7.9521513, -1.9409962, 1.9806430
7: -2.8941207, -0.7644250, -2.8821146, -0.7715940, -1.7703853, 1.7803464
8: 1.9706073, 3.6145763, 1.9785428, 3.6083317, -1.3289495, 1.2919302
9: -8.0629902, -5.5628877, -8.0465965, -5.5890613, -1.9990005, 2.0180087

Time for backsubstitution: 21.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5871
type: B, layer: 1, pos: 5871
type: B, layer: 1, pos: 5856
type: A, layer: 1, pos: 5856
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 4572
type: B, layer: 1, pos: 4572
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 4571
type: B, layer: 1, pos: 5858
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 5801
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 887
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 6139

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 5871

### Candidate
type: B, layer: 1, pos: 5871

### Candidate
type: B, layer: 1, pos: 5856

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0354065, upper bound: 1.0321693
time: 4.27 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0354065, upper bound: 1.0378684
time: 4.05 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -5.3440113, -2.6554108, -5.3505507, -2.6504331, -2.2022476, 2.1930966
1: -6.2944164, -4.2708049, -6.3103294, -4.2538004, -1.7235289, 1.7021554
2: -4.6456294, -2.6445088, -4.6524878, -2.6363714, -1.5702453, 1.5687201
3: -7.8348594, -5.1250367, -7.8560514, -5.1085997, -2.0309668, 2.0277472
4: -11.8021727, -9.0500870, -11.8132725, -9.0366364, -2.3240852, 2.3428202
5: -6.3594251, -4.2419524, -6.3642120, -4.2366343, -1.7208619, 1.7210836
6: -10.4176636, -7.9661722, -10.4398975, -7.9400930, -1.9646792, 1.9500375
7: -2.8767250, -0.7765832, -2.8915026, -0.7670894, -1.7508726, 1.7948871
8: 1.9791789, 3.5923195, 1.9650264, 3.6037345, -1.3212271, 1.3018067
9: -8.0617371, -5.5633993, -8.0710030, -5.5600657, -2.0234461, 2.0343919

Time for backsubstitution: 21.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5871
type: B, layer: 1, pos: 5871
type: A, layer: 1, pos: 5856
type: B, layer: 1, pos: 5856
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 4571
type: B, layer: 1, pos: 4572
type: A, layer: 1, pos: 4572
type: B, layer: 1, pos: 4573
type: A, layer: 1, pos: 4573
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 5858
type: B, layer: 1, pos: 5858
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 5801
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 6139
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 5871

### Candidate
type: B, layer: 1, pos: 5871

### Candidate
type: A, layer: 1, pos: 5856

## Relational analysis of NS_A2_B2_A1_A1

### Relational analysis result of NS_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0308809, upper bound: 1.0322314
time: 4.96 seconds

## Relational analysis of NS_A2_B2_A1_A2

### Relational analysis result of NS_A2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0326922, upper bound: 1.0308246
time: 4.47 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -5.3544474, -2.6463296, -5.3544459, -2.6463120, -2.2213550, 2.2097964
1: -6.3247375, -4.2507954, -6.3247614, -4.2507801, -1.7372932, 1.7369413
2: -4.6537752, -2.6295061, -4.6537747, -2.6294863, -1.5855274, 1.5780208
3: -7.8594122, -5.0937281, -7.8594313, -5.0932169, -2.0665255, 2.0342026
4: -11.8232412, -9.0321960, -11.8232574, -9.0321789, -2.3354921, 2.3706141
5: -6.3656063, -4.2321205, -6.3656034, -4.2321143, -1.7331367, 1.7321396
6: -10.4612913, -7.9368396, -10.4613113, -7.9367981, -1.9686489, 2.0018744
7: -2.8967946, -0.7577980, -2.8968236, -0.7577922, -1.7808175, 1.8026299
8: 1.9638205, 3.6149707, 1.9638200, 3.6149883, -1.3416879, 1.3002307
9: -8.0758667, -5.5572195, -8.0758839, -5.5572219, -2.0442042, 2.0483499

Time for backsubstitution: 22.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5871
type: B, layer: 1, pos: 5871
type: A, layer: 1, pos: 5856
type: B, layer: 1, pos: 5856
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 4572
type: A, layer: 1, pos: 4572
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 4573
type: B, layer: 1, pos: 5858
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 4571
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 5801
type: B, layer: 1, pos: 5801
type: A, layer: 1, pos: 5858
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 887
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 6139

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 5871

### Candidate
type: B, layer: 1, pos: 5871

### Candidate
type: A, layer: 1, pos: 5856

## Relational analysis of NS_A2_B2_A2_A1

### Relational analysis result of NS_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0335957, upper bound: 1.0392968
time: 4.92 seconds

## Relational analysis of NS_A2_B2_A2_A2

### Relational analysis result of NS_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0392947, upper bound: 1.0392968
time: 4.40 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 31.53 seconds
NS_A1_A1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 31.53
Output dim: 7, lower bound: -1.0213511, upper bound: 1.0354063
NS_A1_A1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 31.53
Output dim: 7, lower bound: -1.0270392, upper bound: 1.0354069
NS_A1_A2_B1_B1, status: Status.VERIFIED, split count: 4, time: 31.53
Output dim: 7, lower bound: -1.0214015, upper bound: 1.0308820
NS_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 31.53
Output dim: 7, lower bound: -1.0214015, upper bound: 1.0365806
NS_A1_A2_B2_B1, status: Status.VERIFIED, split count: 4, time: 31.53
Output dim: 7, lower bound: -1.0284669, upper bound: 1.0335970
NS_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 31.53
Output dim: 7, lower bound: -1.0284669, upper bound: 1.0392959
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 31.53
Output dim: 7, lower bound: -1.0354065, upper bound: 1.0321693
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 31.53
Output dim: 7, lower bound: -1.0354065, upper bound: 1.0378684
NS_A2_B2_A1_A1, status: Status.VERIFIED, split count: 4, time: 31.53
Output dim: 7, lower bound: -1.0308809, upper bound: 1.0322314
NS_A2_B2_A1_A2, status: Status.VERIFIED, split count: 4, time: 31.53
Output dim: 7, lower bound: -1.0326922, upper bound: 1.0308246
NS_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 31.53
Output dim: 7, lower bound: -1.0335957, upper bound: 1.0392968
NS_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 31.53
Output dim: 7, lower bound: -1.0392947, upper bound: 1.0392968

## BFS NS instance: NS_A1_A1_A2_A1

### Backsubstitution after applying NS history:
0: -5.3125677, -2.6649876, -5.3399434, -2.6506214, -2.1692090, 2.1818099
1: -6.3047571, -4.2758188, -6.3215151, -4.2611170, -1.6941066, 1.7184720
2: -4.6209526, -2.6547685, -4.6481109, -2.6409826, -1.5426788, 1.5501750
3: -7.8333406, -5.1179647, -7.8482018, -5.0946894, -2.0399613, 2.0022297
4: -11.7743759, -9.0758724, -11.8038177, -9.0376434, -2.3023806, 2.3077188
5: -6.3414760, -4.2544279, -6.3564649, -4.2350316, -1.7059736, 1.6981916
6: -10.4330463, -7.9578347, -10.4510984, -7.9414220, -1.9315124, 1.9768577
7: -2.8535423, -0.7903826, -2.8832579, -0.7645710, -1.7674236, 1.7541637
8: 1.9932022, 3.5937948, 1.9710197, 3.6078167, -1.3142009, 1.2957284
9: -8.0338573, -5.5989943, -8.0599442, -5.5633278, -2.0090313, 1.9925756

Time for backsubstitution: 21.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5871
type: B, layer: 1, pos: 5871
type: B, layer: 1, pos: 5856
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 4572
type: A, layer: 1, pos: 4572
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 5858
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 4571
type: A, layer: 1, pos: 6139
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 5801
type: B, layer: 1, pos: 5801
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 5858
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 887
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 6139

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 5871

### Candidate
type: B, layer: 1, pos: 5871

### Candidate
type: B, layer: 1, pos: 5856

## Relational analysis of NS_A1_A1_A2_A1_B1

### Relational analysis result of NS_A1_A1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0213447, upper bound: 1.0297050
time: 4.30 seconds

## Relational analysis of NS_A1_A1_A2_A1_B2

### Relational analysis result of NS_A1_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0213447, upper bound: 1.0354063
time: 4.58 seconds

## BFS NS instance: NS_A1_A1_A2_A2

### Backsubstitution after applying NS history:
0: -5.3527565, -2.6586075, -5.3399358, -2.6506286, -2.2096839, 2.1895280
1: -6.3187127, -4.2416229, -6.3214946, -4.2611208, -1.7050095, 1.7265608
2: -4.6392498, -2.6258030, -4.6481037, -2.6409841, -1.5637569, 1.5677085
3: -7.8788052, -5.1104984, -7.8481970, -5.0947089, -2.0525098, 2.0093503
4: -11.7864227, -8.9914398, -11.8037882, -9.0376549, -2.3165245, 2.3462219
5: -6.3616304, -4.2502236, -6.3564582, -4.2350311, -1.7399631, 1.7023809
6: -10.4592953, -7.9517908, -10.4510899, -7.9414301, -1.9718328, 1.9827976
7: -2.8649223, -0.7287233, -2.8832328, -0.7645805, -1.7775607, 1.7918901
8: 1.9720068, 3.5962996, 1.9710274, 3.6078143, -1.3285067, 1.2991705
9: -8.0466022, -5.5280113, -8.0599203, -5.5633397, -2.0223484, 2.0430388

Time for backsubstitution: 22.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5871
type: B, layer: 1, pos: 5871
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 5856
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 4572
type: A, layer: 1, pos: 4572
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 4571
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 5801
type: B, layer: 1, pos: 5801
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 887
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 6139

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 5871

### Candidate
type: B, layer: 1, pos: 5871

### Candidate
type: B, layer: 1, pos: 957

## Relational analysis of NS_A1_A1_A2_A2_B1

### Relational analysis result of NS_A1_A1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0270392, upper bound: 1.0302780
time: 4.68 seconds

## Relational analysis of NS_A1_A1_A2_A2_B2

### Relational analysis result of NS_A1_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0270392, upper bound: 1.0354069
time: 4.47 seconds

## BFS NS instance: NS_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -5.3409290, -2.6554403, -5.3813124, -2.6534472, -2.1941090, 2.2263026
1: -6.3055105, -4.2615223, -6.3035040, -4.2414947, -1.7099974, 1.7213316
2: -4.6407056, -2.6447344, -4.6597304, -2.6201825, -1.5704064, 1.5793574
3: -7.8552265, -5.1094418, -7.8776178, -5.1212997, -2.0257812, 2.0395844
4: -11.7987852, -9.0497808, -11.7980595, -8.9697514, -2.3684411, 2.3302135
5: -6.3602619, -4.2378168, -6.3763809, -4.2382283, -1.7196212, 1.7470567
6: -10.4347401, -7.9434462, -10.4410648, -7.9641380, -1.9501390, 1.9898496
7: -2.8690553, -0.7834673, -2.8709025, -0.7174754, -1.8065243, 1.7565539
8: 1.9768867, 3.5901756, 1.9612327, 3.5870829, -1.3031368, 1.3271703
9: -8.0652122, -5.5664897, -8.0645227, -5.4965320, -2.0740094, 2.0288033

Time for backsubstitution: 23.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5871
type: A, layer: 1, pos: 5871
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 5856
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 4571
type: A, layer: 1, pos: 4572
type: B, layer: 1, pos: 4572
type: B, layer: 1, pos: 4573
type: A, layer: 1, pos: 4573
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 5858
type: A, layer: 1, pos: 5858
type: B, layer: 1, pos: 5801
type: A, layer: 1, pos: 6139
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 887
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 5871

### Candidate
type: A, layer: 1, pos: 5871

### Candidate
type: B, layer: 1, pos: 957

## Relational analysis of NS_A1_A2_B1_B2_B1

### Relational analysis result of NS_A1_A2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0214015, upper bound: 1.0314445
time: 4.44 seconds

## Relational analysis of NS_A1_A2_B1_B2_B2

### Relational analysis result of NS_A1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0214015, upper bound: 1.0365805
time: 5.26 seconds

## BFS NS instance: NS_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -5.3448248, -2.6513221, -5.3917079, -2.6443658, -2.2108040, 2.2470798
1: -6.3199425, -4.2585034, -6.3337669, -4.2214866, -1.7392154, 1.7350688
2: -4.6419878, -2.6378489, -4.6679001, -2.6051683, -1.5796969, 1.5948083
3: -7.8586063, -5.0940552, -7.9021578, -5.0900087, -2.0322175, 2.0728178
4: -11.8087664, -9.0453224, -11.8191214, -8.9518747, -2.3938010, 2.3416023
5: -6.3616533, -4.2332983, -6.3825369, -4.2283969, -1.7306852, 1.7594244
6: -10.4561577, -7.9401531, -10.4847107, -7.9347968, -2.0019832, 1.9937470
7: -2.8743761, -0.7741704, -2.8912389, -0.6986954, -1.8143423, 1.7865384
8: 1.9756827, 3.6014290, 1.9458823, 3.6097355, -1.3015611, 1.3476307
9: -8.0700874, -5.5636454, -8.0786457, -5.4903698, -2.0879579, 2.0495572

Time for backsubstitution: 23.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5871
type: A, layer: 1, pos: 5871
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 5856
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 4572
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 4571
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 5801
type: A, layer: 1, pos: 5801
type: B, layer: 1, pos: 5858
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 6139

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 5871

### Candidate
type: A, layer: 1, pos: 5871

### Candidate
type: B, layer: 1, pos: 957

## Relational analysis of NS_A1_A2_B2_B2_B1

### Relational analysis result of NS_A1_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0284669, upper bound: 1.0341652
time: 5.02 seconds

## Relational analysis of NS_A1_A2_B2_B2_B2

### Relational analysis result of NS_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0284669, upper bound: 1.0392959
time: 4.72 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 56.42 + 543.75 = 600.17 seconds
