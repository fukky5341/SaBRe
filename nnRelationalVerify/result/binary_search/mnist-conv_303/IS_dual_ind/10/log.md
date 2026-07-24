## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 1.61837746184
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.8099289, 3.8099289)
1: (-6.7699232, -3.7522836, -6.7699232, -3.7522836, -3.0176396, 3.0176396)
2: (-5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.9518657, 2.9518657)
3: (-8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.8538532, 3.8538537)
4: (-12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487)
5: (-6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1138687)
6: (-10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.5631905, 3.5631905)
7: (-3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.1132982, 3.1132982)
8: (1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3615036, 2.3615036)
9: (-8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.6717334, 3.6717334)

## BASE Result
execution time: IAR + LP analysis = 13.45 + 37.14 = 50.59 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -2.3592026, upper bound: 2.3592013


# Binary Search by BASE starts (time budget: 3549.41 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=3.113298177719116
rel_dist={7: [-1.9085418593697223, 1.908540974116348]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=3.029996871948242
rel_dist={7: [-1.6196747972199932, 1.6196725960818332]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=2.9538488388061523
rel_dist={7: [-1.40751869520538, 1.4075185032663615]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=2.9919233322143555
rel_dist={7: [-1.5160619893642089, 1.516059366795825]}

## Binary Search Result
Binary search time: 208.70 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Individual Split (IS_dual_ind) starts
Time budget: 3340.71 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5871
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 5856
type: A, layer: 1, pos: 4571
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 4682
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 67

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5871

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9706050, upper bound: 1.9960591
time: 4.76 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9962154, upper bound: 1.9962160
time: 5.18 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 10.12 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 10.12
Output dim: 7, lower bound: -1.9706050, upper bound: 1.9960591
IS_A2, status: Status.UNKNOWN, split count: 1, time: 10.12
Output dim: 7, lower bound: -1.9962154, upper bound: 1.9962160

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -5.9552526, -2.1846352, -5.9718370, -2.1695762, -3.6002407, 3.6016364
1: -6.7589445, -3.7738898, -6.7681580, -3.7573121, -2.8810954, 2.8781958
2: -5.0946593, -2.2008746, -5.1117649, -2.1719632, -2.7818146, 2.7684538
3: -8.5150728, -4.5967035, -8.5258198, -4.5930228, -3.4001937, 3.4072657
4: -12.2691593, -8.3755531, -12.3123894, -8.3528271, -3.9163322, 3.9368362
5: -6.8573399, -3.7663674, -6.8686180, -3.7597513, -3.0975885, 3.1022506
6: -10.9017916, -7.3617334, -10.9120665, -7.3533955, -3.3302665, 3.3338857
7: -3.4215825, -0.3953617, -3.4706423, -0.3745701, -3.0470123, 3.0752807
8: 1.5516086, 3.8579321, 1.5354142, 3.8858261, -2.3342175, 2.3225179
9: -8.6768417, -5.0500345, -8.7010517, -5.0382414, -3.5427618, 3.5542684

Time for backsubstitution: 13.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 5871
type: B, layer: 1, pos: 5856
type: B, layer: 1, pos: 4571
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 4572
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 4682
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 67

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 957

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5871

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9706035, upper bound: 1.9706038
time: 4.88 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9706035, upper bound: 1.9960592
time: 4.94 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -5.9742651, -2.1646063, -5.9743538, -2.1644249, -3.6250305, 3.6136327
1: -6.7698288, -3.7524176, -6.7699232, -3.7522836, -2.9099512, 2.9015160
2: -5.1138420, -2.1623573, -5.1139922, -2.1621265, -2.8120284, 2.8074934
3: -8.5302773, -4.5920773, -8.5304804, -4.5920582, -3.4151802, 3.4166303
4: -12.3267860, -8.3508511, -12.3270998, -8.3508511, -3.9759350, 3.9762487
5: -6.8726435, -3.7589469, -6.8727450, -3.7588763, -3.1137671, 3.1137981
6: -10.9138498, -7.3507442, -10.9138498, -7.3506594, -3.3483062, 3.3470078
7: -3.4867694, -0.3735499, -3.4868472, -0.3735490, -3.1132205, 3.1132972
8: 1.5337977, 3.8952694, 1.5337720, 3.8952756, -2.3614779, 2.3614974
9: -8.7084379, -5.0371690, -8.7088528, -5.0371194, -3.5685444, 3.5753784

Time for backsubstitution: 13.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5871
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 5856
type: B, layer: 1, pos: 4571
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 4572
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 4682
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 67

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5871

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9960602, upper bound: 1.9706038
time: 4.92 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9960602, upper bound: 1.9962155
time: 4.99 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 23.97 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 23.97
Output dim: 7, lower bound: -1.9706035, upper bound: 1.9706038
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 23.97
Output dim: 7, lower bound: -1.9706035, upper bound: 1.9960592
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 23.97
Output dim: 7, lower bound: -1.9960602, upper bound: 1.9706038
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 23.97
Output dim: 7, lower bound: -1.9960602, upper bound: 1.9962155

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -5.9552526, -2.1846352, -5.9552526, -2.1846352, -3.5846162, 3.5846152
1: -6.7589445, -3.7738898, -6.7589445, -3.7738898, -2.8663311, 2.8663311
2: -5.0946593, -2.2008746, -5.0946593, -2.2008746, -2.7510004, 2.7510002
3: -8.5150728, -4.5967035, -8.5150728, -4.5967035, -3.3964715, 3.3964710
4: -12.2691593, -8.3755531, -12.2691593, -8.3755531, -3.8936062, 3.8936062
5: -6.8573399, -3.7663674, -6.8573399, -3.7663674, -3.0909724, 3.0909724
6: -10.9017916, -7.3617334, -10.9017916, -7.3617334, -3.3213573, 3.3213558
7: -3.4215825, -0.3953617, -3.4215825, -0.3953617, -3.0262208, 3.0262208
8: 1.5516086, 3.8579321, 1.5516086, 3.8579321, -2.3063235, 2.3063235
9: -8.6768417, -5.0500345, -8.6768417, -5.0500345, -3.5305681, 3.5305686

Time for backsubstitution: 13.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 5856
type: A, layer: 1, pos: 4571
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 4682
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 67

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 957

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5856

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9581418, upper bound: 1.9705751
time: 4.86 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9705593, upper bound: 1.9705755
time: 4.78 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -5.9552526, -2.1846352, -5.9742651, -2.1646063, -3.6052566, 3.6041670
1: -6.7589445, -3.7738898, -6.7698288, -3.7524176, -2.8876877, 2.8784208
2: -5.0946593, -2.2008746, -5.1138420, -2.1623573, -2.7920866, 2.7705922
3: -8.5150728, -4.5967035, -8.5302773, -4.5920773, -3.4010439, 3.4118724
4: -12.2691593, -8.3755531, -12.3267860, -8.3508511, -3.9183083, 3.9512329
5: -6.8573399, -3.7663674, -6.8726435, -3.7589469, -3.0983930, 3.1062760
6: -10.9017916, -7.3617334, -10.9138498, -7.3507442, -3.3331456, 3.3364320
7: -3.4215825, -0.3953617, -3.4867694, -0.3735499, -3.0480325, 3.0914078
8: 1.5516086, 3.8579321, 1.5337977, 3.8952694, -2.3436608, 2.3241343
9: -8.6768417, -5.0500345, -8.7084379, -5.0371690, -3.5439215, 3.5616741

Time for backsubstitution: 13.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 5856
type: A, layer: 1, pos: 4571
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 4682
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 67

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 957

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5856

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9581418, upper bound: 1.9960169
time: 5.00 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9705593, upper bound: 1.9960173
time: 5.05 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -5.9742651, -2.1646063, -5.9552526, -2.1846352, -3.6041665, 3.6052561
1: -6.7698288, -3.7524176, -6.7589445, -3.7738898, -2.8784208, 2.8876877
2: -5.1138420, -2.1623573, -5.0946593, -2.2008746, -2.7705927, 2.7920868
3: -8.5302773, -4.5920773, -8.5150728, -4.5967035, -3.4118714, 3.4010439
4: -12.3267860, -8.3508511, -12.2691593, -8.3755531, -3.9512329, 3.9183083
5: -6.8726435, -3.7589469, -6.8573399, -3.7663674, -3.1062760, 3.0983930
6: -10.9138498, -7.3507442, -10.9017916, -7.3617334, -3.3364329, 3.3331456
7: -3.4867694, -0.3735499, -3.4215825, -0.3953617, -3.0914078, 3.0480325
8: 1.5337977, 3.8952694, 1.5516086, 3.8579321, -2.3241343, 2.3436608
9: -8.7084379, -5.0371690, -8.6768417, -5.0500345, -3.5616741, 3.5439219

Time for backsubstitution: 13.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 5856
type: A, layer: 1, pos: 4571
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 4682
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 67

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 957

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5856

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9835996, upper bound: 1.9705606
time: 5.10 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9960146, upper bound: 1.9705607
time: 5.29 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -5.9742651, -2.1646063, -5.9742651, -2.1646063, -3.6135216, 3.6135221
1: -6.7698288, -3.7524176, -6.7698288, -3.7524176, -2.9098368, 2.9098372
2: -5.1138420, -2.1623573, -5.1138420, -2.1623573, -2.8073435, 2.8073440
3: -8.5302773, -4.5920773, -8.5302773, -4.5920773, -3.4151611, 3.4151616
4: -12.3267860, -8.3508511, -12.3267860, -8.3508511, -3.9759350, 3.9759350
5: -6.8726435, -3.7589469, -6.8726435, -3.7589469, -3.1136966, 3.1136966
6: -10.9138498, -7.3507442, -10.9138498, -7.3507442, -3.3470078, 3.3470073
7: -3.4867694, -0.3735499, -3.4867694, -0.3735499, -3.1132195, 3.1132195
8: 1.5337977, 3.8952694, 1.5337977, 3.8952694, -2.3614717, 2.3614717
9: -8.7084379, -5.0371690, -8.7084379, -5.0371690, -3.5684948, 3.5684948

Time for backsubstitution: 13.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 5856
type: A, layer: 1, pos: 4571
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 4682
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 67

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 957

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5856

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9836007, upper bound: 1.9712253
time: 4.94 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9960158, upper bound: 1.9712252
time: 5.27 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 30.15 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 30.15
Output dim: 7, lower bound: -1.9581418, upper bound: 1.9705751
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 30.15
Output dim: 7, lower bound: -1.9705593, upper bound: 1.9705755
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 30.15
Output dim: 7, lower bound: -1.9581418, upper bound: 1.9960169
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 30.15
Output dim: 7, lower bound: -1.9705593, upper bound: 1.9960173
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 30.15
Output dim: 7, lower bound: -1.9835996, upper bound: 1.9705606
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 30.15
Output dim: 7, lower bound: -1.9960146, upper bound: 1.9705607
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 30.15
Output dim: 7, lower bound: -1.9836007, upper bound: 1.9712253
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 30.15
Output dim: 7, lower bound: -1.9960158, upper bound: 1.9712252

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -5.9494452, -2.1898978, -5.9552526, -2.1846352, -3.5160046, 3.5792866
1: -6.7469740, -3.7779689, -6.7589445, -3.7738898, -2.9147658, 2.8622918
2: -5.0875225, -2.2032886, -5.0946593, -2.2008746, -2.7278152, 2.7478187
3: -8.5097952, -4.6079874, -8.5150728, -4.5967035, -3.3908639, 3.4048648
4: -12.2452774, -8.3839903, -12.2691593, -8.3755531, -3.8697243, 3.8851690
5: -6.8520870, -3.7683582, -6.8573399, -3.7663674, -3.0857196, 3.0889816
6: -10.8963995, -7.3680105, -10.9017916, -7.3617334, -3.3131223, 3.2620492
7: -3.4048233, -0.4021528, -3.4215825, -0.3953617, -3.0094616, 3.0194297
8: 1.5591693, 3.8548236, 1.5516086, 3.8579321, -2.2987628, 2.3032150
9: -8.6583109, -5.0587196, -8.6768417, -5.0500345, -3.5135899, 3.5221682

Time for backsubstitution: 13.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 5856
type: B, layer: 1, pos: 4571
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 4572
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 4682
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 67

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 957

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5856

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9581542, upper bound: 1.9581527
time: 4.82 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9581526, upper bound: 1.9705752
time: 5.06 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -6.0009785, -2.1764703, -5.9552460, -2.1846414, -3.6272087, 3.5934019
1: -6.7710133, -3.7226748, -6.7589273, -3.7738938, -2.8788471, 2.9097545
2: -5.1113610, -2.1590266, -5.0946522, -2.2008774, -2.7728109, 2.7869048
3: -8.5669022, -4.5918231, -8.5150671, -4.5967226, -3.4384279, 3.3997431
4: -12.2802420, -8.2786083, -12.2691307, -8.3755627, -3.9046793, 3.9905224
5: -6.8823776, -3.7586589, -6.8573341, -3.7663674, -3.1160102, 3.0986753
6: -10.9343138, -7.3556423, -10.9017859, -7.3617415, -3.3603110, 3.3261132
7: -3.4393106, -0.3256223, -3.4215577, -0.3953698, -3.0439408, 3.0959353
8: 1.5276542, 3.8648839, 1.5516181, 3.8579292, -2.3302751, 2.3132658
9: -8.6947823, -4.9729385, -8.6768217, -5.0500445, -3.5435543, 3.5945454

Time for backsubstitution: 13.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 4571
type: B, layer: 1, pos: 5856
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 4572
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 4682
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 67

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 957

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4571

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9564280, upper bound: 1.9670167
time: 4.89 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9705510, upper bound: 1.9705531
time: 5.25 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -5.9494452, -2.1898978, -5.9742651, -2.1646063, -3.5367694, 3.5988379
1: -6.7469740, -3.7779689, -6.7698288, -3.7524176, -2.9357262, 2.8743815
2: -5.0875225, -2.2032886, -5.1138420, -2.1623573, -2.7689490, 2.7674112
3: -8.5097952, -4.6079874, -8.5302773, -4.5920773, -3.3954363, 3.4202204
4: -12.2452774, -8.3839903, -12.3267860, -8.3508511, -3.8944263, 3.9427958
5: -6.8520870, -3.7683582, -6.8726435, -3.7589469, -3.0931401, 3.1042852
6: -10.8963995, -7.3680105, -10.9138498, -7.3507442, -3.3249106, 3.2770603
7: -3.4048233, -0.4021528, -3.4867694, -0.3735499, -3.0312734, 3.0846167
8: 1.5591693, 3.8548236, 1.5337977, 3.8952694, -2.3361001, 2.3210258
9: -8.6583109, -5.0587196, -8.7084379, -5.0371690, -3.5267849, 3.5532742

Time for backsubstitution: 13.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 5856
type: B, layer: 1, pos: 4571
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 4572
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 4682
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 67

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 957

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5856

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9581380, upper bound: 1.9835960
time: 4.86 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9581380, upper bound: 1.9960158
time: 5.09 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -6.0009785, -2.1764703, -5.9742589, -2.1646137, -3.6385670, 3.6129537
1: -6.7710133, -3.7226748, -6.7698112, -3.7524228, -2.9002032, 2.9193068
2: -5.1113610, -2.1590266, -5.1138358, -2.1623607, -2.8138967, 2.7989554
3: -8.5669022, -4.5918231, -8.5302715, -4.5920963, -3.4430509, 3.4151449
4: -12.2802420, -8.2786083, -12.3267555, -8.3508635, -3.9293785, 4.0481472
5: -6.8823776, -3.7586589, -6.8726387, -3.7589498, -3.1234279, 3.1139798
6: -10.9343138, -7.3556423, -10.9138441, -7.3507538, -3.3721004, 3.3411899
7: -3.4393106, -0.3256223, -3.4867451, -0.3735583, -3.0657523, 3.1611228
8: 1.5276542, 3.8648839, 1.5338068, 3.8952656, -2.3676114, 2.3310771
9: -8.6947823, -4.9729385, -8.7084160, -5.0371804, -3.5569077, 3.6111407

Time for backsubstitution: 13.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 4571
type: B, layer: 1, pos: 5856
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 4572
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 4682
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 67

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 957

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4571

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9564133, upper bound: 1.9924607
time: 4.74 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9705363, upper bound: 1.9959935
time: 5.43 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -5.9681950, -2.1698546, -5.9552526, -2.1846352, -3.5976357, 3.5999413
1: -6.7575979, -3.7564816, -6.7589445, -3.7738898, -2.8655872, 2.8836474
2: -5.1066227, -2.1647809, -5.0946593, -2.2008746, -2.7602854, 2.7889001
3: -8.5250816, -4.6035018, -8.5150728, -4.5967035, -3.4063668, 3.3878546
4: -12.3025961, -8.3592844, -12.2691593, -8.3755531, -3.9270430, 3.9098749
5: -6.8673806, -3.7611415, -6.8573399, -3.7663674, -3.1010132, 3.0961983
6: -10.9084702, -7.3571653, -10.9017916, -7.3617334, -3.3282008, 3.3249698
7: -3.4698603, -0.3803453, -3.4215825, -0.3953617, -3.0744987, 3.0412371
8: 1.5413876, 3.8920135, 1.5516086, 3.8579321, -2.3165445, 2.3404050
9: -8.6896601, -5.0458684, -8.6768417, -5.0500345, -3.5418329, 3.5355163

Time for backsubstitution: 13.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 5856
type: B, layer: 1, pos: 4571
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 4572
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 4682
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 67

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 957

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9748568, upper bound: 1.9495361
time: 5.01 seconds

## Relational analysis of IS_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5856

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9835958, upper bound: 1.9581382
time: 4.98 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9835958, upper bound: 1.9705606
time: 4.87 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -6.0201087, -2.1565261, -5.9552460, -2.1846414, -3.6408625, 3.6139612
1: -6.7817707, -3.7016120, -6.7589273, -3.7738938, -2.8908072, 2.9248970
2: -5.1305733, -2.1205502, -5.0946522, -2.2008774, -2.7924509, 2.8117313
3: -8.5820208, -4.5872240, -8.5150671, -4.5967226, -3.4548492, 3.4042912
4: -12.3376617, -8.2538290, -12.2691307, -8.3755627, -3.9620991, 4.0153017
5: -6.8975677, -3.7511153, -6.8573341, -3.7663674, -3.1312003, 3.1062188
6: -10.9464283, -7.3447227, -10.9017859, -7.3617415, -3.3754516, 3.3377962
7: -3.5040321, -0.3036997, -3.4215577, -0.3953698, -3.1086624, 3.1178579
8: 1.5097876, 3.9020305, 1.5516181, 3.8579292, -2.3481417, 2.3504124
9: -8.7260609, -4.9600224, -8.6768217, -5.0500445, -3.5744495, 3.6008339

Time for backsubstitution: 13.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 4571
type: B, layer: 1, pos: 5856
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 4572
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 4682
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 67

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 957

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9872770, upper bound: 1.9495362
time: 5.35 seconds

## Relational analysis of IS_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4571

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9818721, upper bound: 1.9670021
time: 4.70 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9959915, upper bound: 1.9705384
time: 5.24 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -5.9681950, -2.1698546, -5.9742651, -2.1646063, -3.6069908, 3.6082063
1: -6.7575979, -3.7564816, -6.7698288, -3.7524176, -2.8970022, 2.9057972
2: -5.1066227, -2.1647809, -5.1138420, -2.1623573, -2.7970362, 2.8041582
3: -8.5250816, -4.6035018, -8.5302773, -4.5920773, -3.4095745, 3.4020009
4: -12.3025961, -8.3592844, -12.3267860, -8.3508511, -3.9517450, 3.9675016
5: -6.8673806, -3.7611415, -6.8726435, -3.7589469, -3.1084337, 3.1115019
6: -10.9084702, -7.3571653, -10.9138498, -7.3507442, -3.3387756, 3.3388309
7: -3.4698603, -0.3803453, -3.4867694, -0.3735499, -3.0963104, 3.1064241
8: 1.5413876, 3.8920135, 1.5337977, 3.8952694, -2.3538818, 2.3582158
9: -8.6896601, -5.0458684, -8.7084379, -5.0371690, -3.5486536, 3.5600882

Time for backsubstitution: 13.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 5856
type: B, layer: 1, pos: 4571
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 4572
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 4682
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 67

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 957

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9818879, upper bound: 1.9637020
time: 4.98 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9818879, upper bound: 1.9637520
time: 5.03 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -6.0201087, -2.1565261, -5.9742589, -2.1646137, -3.6535988, 3.6222258
1: -6.7817707, -3.7016120, -6.7698112, -3.7524228, -2.9222221, 2.9487667
2: -5.1305733, -2.1205502, -5.1138358, -2.1623607, -2.8292017, 2.8368776
3: -8.5820208, -4.5872240, -8.5302715, -4.5920963, -3.4605365, 3.4184937
4: -12.3376617, -8.2538290, -12.3267555, -8.3508635, -3.9867983, 4.0729265
5: -6.8975677, -3.7511153, -6.8726387, -3.7589498, -3.1386180, 3.1215234
6: -10.9464283, -7.3447227, -10.9138441, -7.3507538, -3.3860264, 3.3516583
7: -3.5040321, -0.3036997, -3.4867451, -0.3735583, -3.1304739, 3.1830454
8: 1.5097876, 3.9020305, 1.5338068, 3.8952656, -2.3854780, 2.3682237
9: -8.7260609, -4.9600224, -8.7084160, -5.0371804, -3.5812683, 3.6273346

Time for backsubstitution: 13.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 4571
type: B, layer: 1, pos: 5856
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 4572
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 4682
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 67

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 957

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9943086, upper bound: 1.9637018
time: 5.50 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9943093, upper bound: 1.9637520
time: 4.93 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 24.51 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.51
Output dim: 7, lower bound: -1.9581542, upper bound: 1.9581527
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.51
Output dim: 7, lower bound: -1.9581526, upper bound: 1.9705752
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.51
Output dim: 7, lower bound: -1.9564280, upper bound: 1.9670167
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.51
Output dim: 7, lower bound: -1.9705510, upper bound: 1.9705531
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.51
Output dim: 7, lower bound: -1.9581380, upper bound: 1.9835960
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.51
Output dim: 7, lower bound: -1.9581380, upper bound: 1.9960158
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.51
Output dim: 7, lower bound: -1.9564133, upper bound: 1.9924607
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.51
Output dim: 7, lower bound: -1.9705363, upper bound: 1.9959935
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.51
Output dim: 7, lower bound: -1.9835958, upper bound: 1.9581382
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.51
Output dim: 7, lower bound: -1.9835958, upper bound: 1.9705606
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.51
Output dim: 7, lower bound: -1.9818721, upper bound: 1.9670021
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.51
Output dim: 7, lower bound: -1.9959915, upper bound: 1.9705384
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.51
Output dim: 7, lower bound: -1.9818879, upper bound: 1.9637020
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.51
Output dim: 7, lower bound: -1.9818879, upper bound: 1.9637520
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.51
Output dim: 7, lower bound: -1.9943086, upper bound: 1.9637018
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.51
Output dim: 7, lower bound: -1.9943093, upper bound: 1.9637520

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -5.9494452, -2.1898978, -5.9494452, -2.1898978, -3.5106783, 3.5106778
1: -6.7469740, -3.7779689, -6.7469740, -3.7779689, -2.9106784, 2.9106789
2: -5.0875225, -2.2032886, -5.0875225, -2.2032886, -2.7248721, 2.7248726
3: -8.5097952, -4.6079874, -8.5097952, -4.6079874, -3.3987603, 3.3987598
4: -12.2452774, -8.3839903, -12.2452774, -8.3839903, -3.8612871, 3.8612871
5: -6.8520870, -3.7683582, -6.8520870, -3.7683582, -3.0837288, 3.0837288
6: -10.8963995, -7.3680105, -10.8963995, -7.3680105, -3.2530942, 3.2530940
7: -3.4048233, -0.4021528, -3.4048233, -0.4021528, -3.0026705, 3.0026705
8: 1.5591693, 3.8548236, 1.5591693, 3.8548236, -2.2956543, 2.2956543
9: -8.6583109, -5.0587196, -8.6583109, -5.0587196, -3.5044708, 3.5044708

Time for backsubstitution: 13.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 4571
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 4682
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 67

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 957

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4571

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9545992, upper bound: 1.9440065
time: 7.92 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9581341, upper bound: 1.9581299
time: 5.03 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -5.9494452, -2.1898978, -6.0009785, -2.1764703, -3.5250854, 3.6218746
1: -6.7469740, -3.7779689, -6.7710133, -3.7226748, -2.9577875, 2.8748126
2: -5.0875225, -2.2032886, -5.1113610, -2.1590266, -2.7628322, 2.7683434
3: -8.5097952, -4.6079874, -8.5669022, -4.5918231, -3.3941422, 3.4433293
4: -12.2452774, -8.3839903, -12.2802420, -8.2786083, -3.9666691, 3.8962517
5: -6.8520870, -3.7683582, -6.8823776, -3.7586589, -3.0934281, 3.1140194
6: -10.8963995, -7.3680105, -10.9343138, -7.3556423, -3.3178873, 3.2891116
7: -3.4048233, -0.4021528, -3.4393106, -0.3256223, -3.0792010, 3.0371578
8: 1.5591693, 3.8548236, 1.5276542, 3.8648839, -2.3057146, 2.3271694
9: -8.6583109, -5.0587196, -8.6947823, -4.9729385, -3.5656457, 3.5351624

Time for backsubstitution: 14.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 4571
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 4682
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 67

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 957

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4571

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9545992, upper bound: 1.9564287
time: 10.88 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9581341, upper bound: 1.9705524
time: 5.76 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -5.9966583, -2.1804352, -5.9379902, -2.2013044, -3.6079397, 3.5019112
1: -6.7580976, -3.7251019, -6.7105489, -3.8027668, -2.8366914, 2.8801551
2: -5.1083694, -2.1652718, -5.0755644, -2.2253835, -2.7572169, 2.7420216
3: -8.5629892, -4.6045918, -8.4741344, -4.6418147, -3.4119306, 3.3699698
4: -12.2698345, -8.2830391, -12.2319221, -8.4041862, -3.8656483, 3.9488831
5: -6.8800888, -3.7649212, -6.8439608, -3.7901559, -3.0899329, 3.0790396
6: -10.9164572, -7.3586245, -10.8370113, -7.4052410, -3.2483797, 3.2380958
7: -3.4335485, -0.3338060, -3.3861711, -0.4248905, -3.0086579, 3.0516958
8: 1.5312691, 3.8559051, 1.5829701, 3.8261757, -2.2949066, 2.2729349
9: -8.6900129, -4.9759440, -8.6495295, -5.0613375, -3.5230083, 3.5211554

Time for backsubstitution: 14.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 4571
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 4682
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 67

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 957

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 565

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9489483, upper bound: 1.9670061
time: 5.04 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9564171, upper bound: 1.9670061
time: 5.24 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -6.0009770, -2.1764741, -5.9552317, -2.1846645, -3.6273947, 3.5933785
1: -6.7709994, -3.7226770, -6.7588558, -3.7739043, -2.8788257, 2.8879766
2: -5.1113591, -2.1590295, -5.0946393, -2.2009029, -2.7709379, 2.7821262
3: -8.5668993, -4.5918360, -8.5150518, -4.5967989, -3.4062099, 3.3997116
4: -12.2802353, -8.2786121, -12.2690773, -8.3755770, -3.9046583, 3.9904652
5: -6.8823767, -3.7586646, -6.8573265, -3.7663980, -3.1159787, 3.0986619
6: -10.9342947, -7.3556433, -10.9016943, -7.3617525, -3.3444557, 3.2780800
7: -3.4393048, -0.3256290, -3.4215353, -0.3954053, -3.0438995, 3.0959063
8: 1.5276575, 3.8648767, 1.5516291, 3.8578887, -2.3302312, 2.3132477
9: -8.6947775, -4.9729414, -8.6768026, -5.0500536, -3.5449066, 3.5945268

Time for backsubstitution: 14.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 4571
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 4682
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 67

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 957

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4571

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9670154, upper bound: 1.9564290
time: 5.23 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9670155, upper bound: 1.9564287
time: 5.50 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -5.9494452, -2.1898978, -5.9681950, -2.1698546, -3.5314574, 3.5923061
1: -6.7469740, -3.7779689, -6.7575979, -3.7564816, -2.9316430, 2.8615479
2: -5.0875225, -2.2032886, -5.1066227, -2.1647809, -2.7659988, 2.7571030
3: -8.5097952, -4.6079874, -8.5250816, -4.6035018, -3.3822470, 3.4142041
4: -12.2452774, -8.3839903, -12.3025961, -8.3592844, -3.8859930, 3.9186058
5: -6.8520870, -3.7683582, -6.8673806, -3.7611415, -3.0909455, 3.0990224
6: -10.8963995, -7.3680105, -10.9084702, -7.3571653, -3.3167357, 3.2681108
7: -3.4048233, -0.4021528, -3.4698603, -0.3803453, -3.0244780, 3.0677075
8: 1.5591693, 3.8548236, 1.5413876, 3.8920135, -2.3328443, 2.3134360
9: -8.6583109, -5.0587196, -8.6896601, -5.0458684, -3.5176630, 3.5334325

Time for backsubstitution: 14.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 4571
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 4682
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 67

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 957

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9371244, upper bound: 1.9748555
time: 4.93 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4571

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9545846, upper bound: 1.9694495
time: 11.33 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9581195, upper bound: 1.9835732
time: 5.19 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -5.9494452, -2.1898978, -6.0201087, -2.1565261, -3.5457444, 3.6355278
1: -6.7469740, -3.7779689, -6.7817707, -3.7016120, -2.9724684, 2.8867726
2: -5.0875225, -2.2032886, -5.1305733, -2.1205502, -2.7876868, 2.7879846
3: -8.5097952, -4.6079874, -8.5820208, -4.5872240, -3.3986912, 3.4597621
4: -12.2452774, -8.3839903, -12.3376617, -8.2538290, -3.9914484, 3.9536715
5: -6.8520870, -3.7683582, -6.8975677, -3.7511153, -3.1009717, 3.1292095
6: -10.8963995, -7.3680105, -10.9464283, -7.3447227, -3.3295698, 3.3041656
7: -3.4048233, -0.4021528, -3.5040321, -0.3036997, -3.1011236, 3.1018794
8: 1.5591693, 3.8548236, 1.5097876, 3.9020305, -2.3428612, 2.3450360
9: -8.6583109, -5.0587196, -8.7260609, -4.9600224, -3.5717649, 3.5660577

Time for backsubstitution: 14.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 4571
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 4682
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 67

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 957

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9371244, upper bound: 1.9872784
time: 5.02 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4571

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9545846, upper bound: 1.9818742
time: 6.63 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9581195, upper bound: 1.9959928
time: 5.01 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -5.9966583, -2.1804352, -5.9568195, -2.1812582, -3.6193180, 3.5847831
1: -6.7580976, -3.7251019, -6.7213931, -3.7812777, -2.8580570, 2.8474123
2: -5.1083694, -2.1652718, -5.0946937, -2.1868761, -2.7983036, 2.7674978
3: -8.5629892, -4.6045918, -8.4895735, -4.6372705, -3.3935881, 3.3855410
4: -12.2698345, -8.2830391, -12.2893286, -8.3794374, -3.8903971, 4.0062895
5: -6.8800888, -3.7649212, -6.8592329, -3.7828870, -3.0972018, 3.0943117
6: -10.9164572, -7.3586245, -10.8490639, -7.3943548, -3.3048668, 3.2533326
7: -3.4335485, -0.3338060, -3.4512210, -0.4031141, -3.0304344, 3.0619237
8: 1.5312691, 3.8559051, 1.5651126, 3.8634233, -2.3321543, 2.2907925
9: -8.6900129, -4.9759440, -8.6809568, -5.0484581, -3.5363750, 3.5560656

Time for backsubstitution: 14.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 4571
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 4682
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 67

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 957

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9354110, upper bound: 1.9837208
time: 5.46 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 565

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9489330, upper bound: 1.9924500
time: 5.43 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9564030, upper bound: 1.9924499
time: 5.20 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -6.0009770, -2.1764741, -5.9742451, -2.1646354, -3.6387548, 3.6129308
1: -6.7709994, -3.7226770, -6.7697377, -3.7524319, -2.9001808, 2.8975291
2: -5.1113591, -2.1590295, -5.1138215, -2.1623878, -2.8120217, 2.7941468
3: -8.5668993, -4.5918360, -8.5302572, -4.5921721, -3.4108315, 3.4151139
4: -12.2802353, -8.2786121, -12.3267021, -8.3508778, -3.9293575, 4.0480900
5: -6.8823767, -3.7586646, -6.8726292, -3.7589779, -3.1233988, 3.1139646
6: -10.9342947, -7.3556433, -10.9137535, -7.3507643, -3.3511310, 3.2931571
7: -3.4393048, -0.3256290, -3.4867220, -0.3735936, -3.0657113, 3.1509719
8: 1.5276575, 3.8648767, 1.5338182, 3.8952250, -2.3675675, 2.3310585
9: -8.6947775, -4.9729414, -8.7083979, -5.0371885, -3.5582581, 3.6111226

Time for backsubstitution: 14.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 4571
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 4682
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 67

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 957

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9495121, upper bound: 1.9872560
time: 6.68 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4571

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9670008, upper bound: 1.9818731
time: 7.04 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9670008, upper bound: 1.9818730
time: 5.61 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -5.9681950, -2.1698546, -5.9494452, -2.1898978, -3.5923061, 3.5314584
1: -6.7575979, -3.7564816, -6.7469740, -3.7779689, -2.8615479, 2.9316430
2: -5.1066227, -2.1647809, -5.0875225, -2.2032886, -2.7571034, 2.7659993
3: -8.5250816, -4.6035018, -8.5097952, -4.6079874, -3.4142041, 3.3822474
4: -12.3025961, -8.3592844, -12.2452774, -8.3839903, -3.9186058, 3.8859930
5: -6.8673806, -3.7611415, -6.8520870, -3.7683582, -3.0990224, 3.0909455
6: -10.9084702, -7.3571653, -10.8963995, -7.3680105, -3.2681108, 3.3167357
7: -3.4698603, -0.3803453, -3.4048233, -0.4021528, -3.0677075, 3.0244780
8: 1.5413876, 3.8920135, 1.5591693, 3.8548236, -2.3134360, 2.3328443
9: -8.6896601, -5.0458684, -8.6583109, -5.0587196, -3.5334320, 3.5176620

Time for backsubstitution: 14.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 4571
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 4682
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 67

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 957

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4571

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9800409, upper bound: 1.9439915
time: 7.88 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9835773, upper bound: 1.9581153
time: 5.11 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -5.9681950, -2.1698546, -6.0009785, -2.1764703, -3.6063013, 3.6332359
1: -6.7575979, -3.7564816, -6.7710133, -3.7226748, -2.9058552, 2.8961682
2: -5.1066227, -2.1647809, -5.1113610, -2.1590266, -2.7886343, 2.8094244
3: -8.5250816, -4.6035018, -8.5669022, -4.5918231, -3.4096460, 3.4278893
4: -12.3025961, -8.3592844, -12.2802420, -8.2786083, -4.0239878, 3.9209576
5: -6.8673806, -3.7611415, -6.8823776, -3.7586589, -3.1087217, 3.1212361
6: -10.9084702, -7.3571653, -10.9343138, -7.3556423, -3.3329678, 3.3536496
7: -3.4698603, -0.3803453, -3.4393106, -0.3256223, -3.1403008, 3.0589652
8: 1.5413876, 3.8920135, 1.5276542, 3.8648839, -2.3234963, 2.3643594
9: -8.6896601, -5.0458684, -8.6947823, -4.9729385, -3.5893736, 3.5485096

Time for backsubstitution: 14.01 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=3.113298177719116
rel_dist={7: [-1.996222388869347, 1.996221151991417]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5871
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 5856
type: A, layer: 1, pos: 4571
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 4682
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 67

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5871

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7062921, upper bound: 1.7205256
time: 4.95 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7205243, upper bound: 1.7205255
time: 4.96 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 10.10 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 10.10
Output dim: 7, lower bound: -1.7062921, upper bound: 1.7205256
IS_A2, status: Status.UNKNOWN, split count: 1, time: 10.10
Output dim: 7, lower bound: -1.7205243, upper bound: 1.7205255

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -5.9552526, -2.1846352, -5.9701324, -2.1730092, -3.4367199, 3.4398122
1: -6.7589445, -3.7738898, -6.7669725, -3.7606611, -2.7170634, 2.7173131
2: -5.0946593, -2.2008746, -5.1102695, -2.1785045, -2.6418662, 2.6340537
3: -8.5150728, -4.5967035, -8.5238657, -4.5936704, -3.1372886, 3.1427922
4: -12.2691593, -8.3755531, -12.3025856, -8.3541584, -3.9150009, 3.9270325
5: -6.8573399, -3.7663674, -6.8664522, -3.7603426, -3.0839148, 3.0880461
6: -10.9017916, -7.3617334, -10.9108572, -7.3552151, -3.1198311, 3.1236944
7: -3.4215825, -0.3953617, -3.4598584, -0.3752573, -3.0002794, 3.0097721
8: 1.5516086, 3.8579321, 1.5365157, 3.8795328, -2.3184042, 2.3125205
9: -8.6768417, -5.0500345, -8.6958570, -5.0389996, -3.3533363, 3.3604975

Time for backsubstitution: 13.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 5856
type: B, layer: 1, pos: 4571
type: B, layer: 1, pos: 5871
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 4572
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 4682
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 67

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 957

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5856

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7062713, upper bound: 1.7101994
time: 5.34 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7062695, upper bound: 1.7205039
time: 4.82 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -5.9742651, -2.1646063, -5.9743400, -2.1644549, -3.4649587, 3.4515877
1: -6.7698288, -3.7524176, -6.7699060, -3.7523069, -2.7493949, 2.7416573
2: -5.1138420, -2.1623573, -5.1139674, -2.1621668, -2.6790953, 2.6738310
3: -8.5302773, -4.5920773, -8.5304432, -4.5920615, -3.1519866, 3.1542759
4: -12.3267860, -8.3508511, -12.3270483, -8.3508530, -3.9759331, 3.9761972
5: -6.8726435, -3.7589469, -6.8727283, -3.7588882, -3.1011333, 3.0991731
6: -10.9138498, -7.3507442, -10.9138508, -7.3506746, -3.1398234, 3.1383245
7: -3.4867694, -0.3735499, -3.4868333, -0.3735499, -3.0639634, 3.0628903
8: 1.5337977, 3.8952694, 1.5337758, 3.8952756, -2.3529594, 2.3333423
9: -8.7084379, -5.0371690, -8.7087831, -5.0371284, -3.3787785, 3.3867145

Time for backsubstitution: 13.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 5871
type: B, layer: 1, pos: 5856
type: B, layer: 1, pos: 4571
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 4572
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 4682
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 67

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 957

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7186339, upper bound: 1.7009012
time: 5.22 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7186339, upper bound: 1.7186349
time: 5.15 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 24.47 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 24.47
Output dim: 7, lower bound: -1.7062713, upper bound: 1.7101994
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 24.47
Output dim: 7, lower bound: -1.7062695, upper bound: 1.7205039
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 24.47
Output dim: 7, lower bound: -1.7186339, upper bound: 1.7009012
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 24.47
Output dim: 7, lower bound: -1.7186339, upper bound: 1.7186349

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -5.9543357, -2.1854532, -5.9640632, -2.1782594, -3.4304171, 3.4324546
1: -6.7570376, -3.7745061, -6.7547398, -3.7647448, -2.7110124, 2.7038646
2: -5.0935397, -2.2012439, -5.1030531, -2.1809268, -2.6370673, 2.6232808
3: -8.5142803, -4.5984869, -8.5185862, -4.6050959, -3.1232319, 3.1351385
4: -12.2653856, -8.3768091, -12.2783880, -8.3625889, -3.9027967, 3.9015789
5: -6.8565478, -3.7667024, -6.8611879, -3.7625272, -3.0773129, 3.0789952
6: -10.9009657, -7.3627381, -10.9054680, -7.3616419, -3.1103983, 3.1141820
7: -3.4189389, -0.3963759, -3.4429333, -0.3820505, -2.9892454, 2.9874432
8: 1.5527539, 3.8574243, 1.5440946, 3.8762736, -2.3144326, 2.3055112
9: -8.6739101, -5.0513320, -8.6770668, -5.0476909, -3.3418350, 3.3393917

Time for backsubstitution: 13.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 5856
type: A, layer: 1, pos: 4571
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 4682
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 67

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 957

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5856

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6959653, upper bound: 1.7101964
time: 4.63 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6959653, upper bound: 1.7101979
time: 5.70 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -5.9552431, -2.1846449, -6.0159426, -2.1649127, -3.4452572, 3.4780936
1: -6.7589154, -3.7738986, -6.7789707, -3.7096636, -2.7564435, 2.7297504
2: -5.0946484, -2.2008791, -5.1269827, -2.1366737, -2.6667671, 2.6540501
3: -8.5150623, -4.5967340, -8.5757151, -4.5888000, -3.1405420, 3.1838744
4: -12.2691097, -8.3755693, -12.3135099, -8.2571526, -3.9800272, 3.9379406
5: -6.8573313, -3.7663693, -6.8915577, -3.7525537, -3.0880747, 3.1251884
6: -10.9017811, -7.3617449, -10.9434185, -7.3491688, -3.1245098, 3.1614976
7: -3.4215407, -0.3953750, -3.4772575, -0.3054371, -3.0337467, 3.0240622
8: 1.5516233, 3.8579264, 1.5125241, 3.8863444, -2.3268363, 2.3378274
9: -8.6768131, -5.0500517, -8.7135534, -4.9618597, -3.4102526, 3.3733134

Time for backsubstitution: 14.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 4571
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 5856
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 4682
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 67

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 957

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4571

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7007326, upper bound: 1.7088108
time: 5.42 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7062536, upper bound: 1.7204873
time: 5.69 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -5.9727817, -2.1679225, -5.9595032, -2.1719968, -3.4556866, 3.4331455
1: -6.7687745, -3.7567942, -6.7618513, -3.7643504, -2.7365575, 2.7287600
2: -5.1121621, -2.1680045, -5.0960093, -2.1744900, -2.6650696, 2.6496036
3: -8.5286961, -4.5929360, -8.5266094, -4.5946889, -3.1474142, 3.1503696
4: -12.3170376, -8.3520889, -12.3058577, -8.3716679, -3.9453697, 3.9537687
5: -6.8702850, -3.7593293, -6.8663850, -3.7638378, -3.0969925, 3.0861115
6: -10.9125500, -7.3529501, -10.9043093, -7.3559141, -3.1330500, 3.1257138
7: -3.4728508, -0.3741739, -3.4550014, -0.3990445, -3.0236454, 3.0305696
8: 1.5347142, 3.8863583, 1.5520363, 3.8760552, -2.3325653, 2.3054433
9: -8.7043800, -5.0380135, -8.6979561, -5.0467739, -3.3649111, 3.3753304

Time for backsubstitution: 13.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5856
type: A, layer: 1, pos: 4571
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 4682
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 67

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5856

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7083017, upper bound: 1.7008785
time: 5.71 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7186123, upper bound: 1.7008787
time: 5.30 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -5.9742270, -2.1646848, -5.9742808, -2.1645670, -3.4583507, 3.4514585
1: -6.7698116, -3.7524509, -6.7698865, -3.7523694, -2.7462907, 2.7416034
2: -5.1138415, -2.1624398, -5.1139641, -2.1622853, -2.6687794, 2.6737738
3: -8.5302191, -4.5927296, -8.5303440, -4.5934405, -3.1507149, 3.1535411
4: -12.3266735, -8.3508739, -12.3268986, -8.3508968, -3.9757767, 3.9760246
5: -6.8726244, -3.7589583, -6.8726978, -3.7588992, -3.0986133, 3.0939994
6: -10.9138498, -7.3508105, -10.9138479, -7.3507948, -3.1381454, 3.1382716
7: -3.4867239, -0.3735509, -3.4867468, -0.3735504, -3.0502777, 3.0455632
8: 1.5338092, 3.8952651, 1.5337973, 3.8952651, -2.3358247, 2.3333113
9: -8.7083483, -5.0371828, -8.7086391, -5.0371447, -3.3786545, 3.3838406

Time for backsubstitution: 14.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5856
type: A, layer: 1, pos: 4571
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 4682
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 67

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5856

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7083018, upper bound: 1.7186106
time: 5.33 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7186123, upper bound: 1.7186144
time: 5.09 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 24.71 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 24.71
Output dim: 7, lower bound: -1.6959653, upper bound: 1.7101964
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 24.71
Output dim: 7, lower bound: -1.6959653, upper bound: 1.7101979
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 24.71
Output dim: 7, lower bound: -1.7007326, upper bound: 1.7088108
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 24.71
Output dim: 7, lower bound: -1.7062536, upper bound: 1.7204873
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 24.71
Output dim: 7, lower bound: -1.7083017, upper bound: 1.7008785
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 24.71
Output dim: 7, lower bound: -1.7186123, upper bound: 1.7008787
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 24.71
Output dim: 7, lower bound: -1.7083018, upper bound: 1.7186106
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 24.71
Output dim: 7, lower bound: -1.7186123, upper bound: 1.7186144

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -5.9494452, -2.1898978, -5.9640632, -2.1782594, -3.3546104, 3.4279542
1: -6.7469740, -3.7779689, -6.7547398, -3.7647448, -2.7478557, 2.7004347
2: -5.0875225, -2.2032886, -5.1030531, -2.1809268, -2.6209927, 2.6205819
3: -8.5097952, -4.6079874, -8.5185862, -4.6050959, -3.1184931, 3.1337762
4: -12.2452774, -8.3839903, -12.2783880, -8.3625889, -3.8826885, 3.8943977
5: -6.8520870, -3.7683582, -6.8611879, -3.7625272, -3.0703607, 3.0592313
6: -10.8963995, -7.3680105, -10.9054680, -7.3616419, -3.1034145, 3.0575285
7: -3.4048233, -0.4021528, -3.4429333, -0.3820505, -3.0066519, 2.9809284
8: 1.5591693, 3.8548236, 1.5440946, 3.8762736, -2.3088942, 2.2616050
9: -8.6583109, -5.0587196, -8.6770668, -5.0476909, -3.3155365, 3.3322482

Time for backsubstitution: 14.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 4571
type: B, layer: 1, pos: 5871
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 4572
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 4682
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 67

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 957

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4571

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6843207, upper bound: 1.7047424
time: 5.14 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6959471, upper bound: 1.7101827
time: 5.23 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -6.0009785, -2.1764703, -5.9640632, -2.1782594, -3.4644966, 3.4419494
1: -6.7710133, -3.7226748, -6.7547398, -3.7647448, -2.7255330, 2.7441199
2: -5.1113610, -2.1590266, -5.1030531, -2.1809268, -2.6592045, 2.6576304
3: -8.5669022, -4.5918231, -8.5185862, -4.6050959, -3.1642714, 3.1404734
4: -12.2802420, -8.2786083, -12.2783880, -8.3625889, -3.9176531, 3.9509878
5: -6.8823776, -3.7586589, -6.8611879, -3.7625272, -3.1022725, 3.0839162
6: -10.9343138, -7.3556423, -10.9054680, -7.3616419, -3.1403284, 3.1202269
7: -3.4393106, -0.3256223, -3.4429333, -0.3820505, -3.0074863, 3.0115793
8: 1.5276542, 3.8648839, 1.5440946, 3.8762736, -2.3400776, 2.3145649
9: -8.6947823, -4.9729385, -8.6770668, -5.0476909, -3.3579254, 3.3884420

Time for backsubstitution: 14.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 4571
type: B, layer: 1, pos: 5871
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 4572
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 4682
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 67

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 957

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4571

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6843207, upper bound: 1.7047413
time: 5.20 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6959470, upper bound: 1.7101842
time: 5.61 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -5.9379878, -2.2013087, -6.0086079, -2.1717167, -3.3478489, 3.4558506
1: -6.7105370, -3.8027706, -6.7568483, -3.7137742, -2.7107983, 2.6786885
2: -5.0755606, -2.2253871, -5.1219139, -2.1473625, -2.6222601, 2.6348634
3: -8.4741306, -4.6418257, -8.5690346, -4.6107044, -3.0998015, 3.1421556
4: -12.2319050, -8.4041920, -12.2956619, -8.2647419, -3.8804321, 3.8914700
5: -6.8439574, -3.7901576, -6.8876076, -3.7632217, -3.0617838, 2.9552398
6: -10.8370085, -7.4052439, -10.9127874, -7.3542967, -3.0307012, 3.0356333
7: -3.3861537, -0.4248948, -3.4673023, -0.3194709, -2.9473724, 2.9780819
8: 1.5829763, 3.8261752, 1.5186648, 3.8709321, -2.2879558, 2.2583268
9: -8.6495190, -5.0613446, -8.7054634, -4.9669518, -3.3249712, 3.3497329

Time for backsubstitution: 14.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 5871
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 4572
type: B, layer: 1, pos: 4571
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 4682
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 67

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 957

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5871

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7007326, upper bound: 1.6961057
time: 5.46 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7007312, upper bound: 1.7088108
time: 5.46 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -5.9552274, -2.1846676, -6.0159359, -2.1649230, -3.4452267, 3.4781907
1: -6.7588429, -3.7739077, -6.7789373, -3.7096672, -2.7297306, 2.7297080
2: -5.0946350, -2.2009048, -5.1269755, -2.1366844, -2.6619844, 2.6518493
3: -8.5150480, -4.5968103, -8.5757103, -4.5888338, -3.1404891, 3.1458278
4: -12.2690563, -8.3755856, -12.3134842, -8.2571602, -3.9621687, 3.9378986
5: -6.8573217, -3.7663980, -6.8915529, -3.7525666, -3.0889235, 3.1251550
6: -10.9016905, -7.3617578, -10.9433727, -7.3491759, -3.0679970, 3.1428599
7: -3.4215176, -0.3954093, -3.4772477, -0.3054540, -3.0233068, 3.0038803
8: 1.5516357, 3.8578863, 1.5125299, 3.8863258, -2.3189304, 2.3204968
9: -8.6767950, -5.0500617, -8.7135429, -4.9618645, -3.4102325, 3.3740697

Time for backsubstitution: 13.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 5871
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 4572
type: B, layer: 1, pos: 4571
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 4682
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 67

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 957

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5871

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7062536, upper bound: 1.7077806
time: 5.40 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7062536, upper bound: 1.7204873
time: 5.38 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -5.9667215, -2.1731696, -5.9585905, -2.1728144, -3.4483380, 3.4268565
1: -6.7565422, -3.7608659, -6.7599430, -3.7649646, -2.7231145, 2.7227156
2: -5.1049585, -2.1704268, -5.0948911, -2.1748600, -2.6542921, 2.6448073
3: -8.5235214, -4.6043611, -8.5258236, -4.5964732, -3.1397896, 3.1363254
4: -12.2928495, -8.3604889, -12.3020868, -8.3729115, -3.9199381, 3.9415979
5: -6.8650274, -3.7615185, -6.8655901, -3.7641759, -3.0879579, 3.0795002
6: -10.9071693, -7.3593745, -10.9034882, -7.3569174, -3.1235566, 3.1162925
7: -3.4559453, -0.3809419, -3.4523649, -0.4000490, -3.0014906, 3.0191240
8: 1.5422993, 3.8831015, 1.5531840, 3.8755479, -2.3255572, 2.3014765
9: -8.6856022, -5.0466866, -8.6950293, -5.0480652, -3.3438225, 3.3638568

Time for backsubstitution: 13.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5871
type: B, layer: 1, pos: 5856
type: B, layer: 1, pos: 4571
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 4572
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 4682
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 67

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5871

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5856

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7083009, upper bound: 1.6905945
time: 5.75 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7083009, upper bound: 1.7008786
time: 5.85 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -6.0186219, -2.1598220, -5.9594941, -2.1720076, -3.4880056, 3.4415956
1: -6.7807341, -3.7058790, -6.7618208, -3.7643585, -2.7489586, 2.7668223
2: -5.1288476, -2.1261687, -5.0959988, -2.1744943, -2.6850624, 2.6805754
3: -8.5804443, -4.5880842, -8.5265999, -4.5947199, -3.1890359, 3.1536093
4: -12.3279514, -8.2550516, -12.3058138, -8.3716841, -3.9562674, 4.0031214
5: -6.8952007, -3.7515173, -6.8663754, -3.7638402, -3.1313605, 3.0903249
6: -10.9451170, -7.3469205, -10.9042997, -7.3559265, -3.1708770, 3.1303720
7: -3.4901831, -0.3043194, -3.4549623, -0.3990567, -3.0378151, 3.0535455
8: 1.5107126, 3.8931456, 1.5520511, 3.8760495, -2.3578973, 2.3138444
9: -8.7220545, -4.9608555, -8.6979294, -5.0467911, -3.3777208, 3.4251194

Time for backsubstitution: 13.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5871
type: B, layer: 1, pos: 4571
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 5856
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 4572
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 4682
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 67

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5871

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4571

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7069439, upper bound: 1.6953780
time: 5.17 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7185957, upper bound: 1.7008628
time: 12.47 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -5.9681559, -2.1699338, -5.9733639, -2.1653833, -3.4509921, 3.4451585
1: -6.7575812, -3.7565141, -6.7679791, -3.7529852, -2.7328472, 2.7355607
2: -5.1066208, -2.1648626, -5.1128440, -2.1626568, -2.6579804, 2.6689696
3: -8.5250244, -4.6041570, -8.5295639, -4.5952282, -3.1430731, 3.1394982
4: -12.3024797, -8.3593063, -12.3231297, -8.3521557, -3.9503241, 3.9638233
5: -6.8673601, -3.7611520, -6.8719025, -3.7592399, -3.0895596, 3.0873742
6: -10.9084673, -7.3572321, -10.9130249, -7.3517957, -3.1286392, 3.1288457
7: -3.4698131, -0.3803461, -3.4841158, -0.3745666, -3.0280151, 3.0341415
8: 1.5413990, 3.8920078, 1.5349469, 3.8947611, -2.3288081, 2.3293383
9: -8.6895685, -5.0458827, -8.7057152, -5.0384426, -3.3575525, 3.3723445

Time for backsubstitution: 13.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5871
type: B, layer: 1, pos: 5856
type: B, layer: 1, pos: 4571
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 4572
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 4682
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 67

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5871

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5856

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7083008, upper bound: 1.7083009
time: 5.26 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7083008, upper bound: 1.7186104
time: 5.28 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -6.0200686, -2.1566081, -5.9742708, -2.1645787, -3.4885120, 3.4599700
1: -6.7817526, -3.7016451, -6.7698569, -3.7523775, -2.7586746, 2.7762108
2: -5.1305704, -2.1206355, -5.1139522, -2.1622906, -2.6887803, 2.6960673
3: -8.5819588, -4.5878801, -8.5303354, -4.5934734, -3.1929946, 3.1567826
4: -12.3375435, -8.2538500, -12.3268518, -8.3509121, -3.9866314, 4.0157514
5: -6.8975458, -3.7511253, -6.8726883, -3.7589033, -3.1370296, 3.0982409
6: -10.9464283, -7.3447881, -10.9138374, -7.3508072, -3.1759758, 3.1429169
7: -3.5039911, -0.3037000, -3.4867067, -0.3735640, -3.0638928, 3.0685291
8: 1.5097990, 3.9020276, 1.5338125, 3.8952618, -2.3587770, 2.3416722
9: -8.7259655, -4.9600358, -8.7086096, -5.0371614, -3.3914194, 3.4309235

Time for backsubstitution: 13.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5871
type: B, layer: 1, pos: 4571
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 5856
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 4572
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 4682
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 67

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5871

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4571

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7069439, upper bound: 1.7130983
time: 5.54 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7185957, upper bound: 1.7185986
time: 4.94 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 30.21 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 30.21
Output dim: 7, lower bound: -1.6843207, upper bound: 1.7047424
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 30.21
Output dim: 7, lower bound: -1.6959471, upper bound: 1.7101827
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.21
Output dim: 7, lower bound: -1.6843207, upper bound: 1.7047413
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.21
Output dim: 7, lower bound: -1.6959470, upper bound: 1.7101842
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 30.21
Output dim: 7, lower bound: -1.7007326, upper bound: 1.6961057
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 30.21
Output dim: 7, lower bound: -1.7007312, upper bound: 1.7088108
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.21
Output dim: 7, lower bound: -1.7062536, upper bound: 1.7077806
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.21
Output dim: 7, lower bound: -1.7062536, upper bound: 1.7204873
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 30.21
Output dim: 7, lower bound: -1.7083009, upper bound: 1.6905945
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 30.21
Output dim: 7, lower bound: -1.7083009, upper bound: 1.7008786
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.21
Output dim: 7, lower bound: -1.7069439, upper bound: 1.6953780
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.21
Output dim: 7, lower bound: -1.7185957, upper bound: 1.7008628
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 30.21
Output dim: 7, lower bound: -1.7083008, upper bound: 1.7083009
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 30.21
Output dim: 7, lower bound: -1.7083008, upper bound: 1.7186104
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.21
Output dim: 7, lower bound: -1.7069439, upper bound: 1.7130983
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.21
Output dim: 7, lower bound: -1.7185957, upper bound: 1.7185986

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -5.9423604, -2.1967051, -5.9466305, -2.1949186, -3.3309383, 3.4009299
1: -6.7250328, -3.7820764, -6.7062912, -3.7936094, -2.6976147, 2.6217127
2: -5.0825586, -2.2139664, -5.0839181, -2.2054312, -2.6002107, 2.5873709
3: -8.5030870, -4.6297574, -8.4776821, -4.6502581, -3.0658545, 3.0904865
4: -12.2277708, -8.3915892, -12.2409687, -8.3911829, -3.8365879, 3.8103247
5: -6.8481393, -3.7788870, -6.8478389, -3.7864528, -2.9112988, 3.0331788
6: -10.8657780, -7.3729777, -10.8407078, -7.4052458, -3.0270705, 2.9683459
7: -3.3951852, -0.4161656, -3.4073963, -0.4115920, -2.9608531, 2.8774137
8: 1.5653114, 3.8395591, 1.5753970, 3.8444228, -2.2682896, 2.2328451
9: -8.6504536, -5.0638223, -8.6495953, -5.0589695, -3.2912598, 3.2761040

Time for backsubstitution: 13.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 4571
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 4682
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 67

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 957

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 565

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6797122, upper bound: 1.7047343
time: 4.93 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6843152, upper bound: 1.7047342
time: 5.28 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -5.9494381, -2.1899066, -5.9640474, -2.1782813, -3.3545699, 3.4279218
1: -6.7469397, -3.7779722, -6.7546673, -3.7647538, -2.7478151, 2.6762025
2: -5.0875144, -2.2033000, -5.1030397, -2.1809533, -2.6190119, 2.6205530
3: -8.5097857, -4.6080203, -8.5185728, -4.6051707, -3.0808878, 3.1337242
4: -12.2452517, -8.3839970, -12.2783337, -8.3626080, -3.8826437, 3.8943367
5: -6.8520832, -3.7683737, -6.8611784, -3.7625577, -3.0701098, 3.0596790
6: -10.8963547, -7.3680162, -10.9053783, -7.3616533, -3.1033640, 3.0015466
7: -3.4048121, -0.4021688, -3.4429109, -0.3820863, -2.9870167, 2.9704878
8: 1.5591750, 3.8548045, 1.5441084, 3.8762331, -2.2915621, 2.2615755
9: -8.6583033, -5.0587239, -8.6770487, -5.0477009, -3.3160896, 3.3322186

Time for backsubstitution: 13.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 4571
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 4682
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 67

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 957

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 565

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6912121, upper bound: 1.7101745
time: 5.21 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6959415, upper bound: 1.7101746
time: 5.01 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -5.9936528, -2.1832769, -5.9466305, -2.1949186, -3.4422674, 3.4149280
1: -6.7488761, -3.7267828, -6.7062912, -3.7936094, -2.6744065, 2.6683290
2: -5.1062870, -2.1697125, -5.0839181, -2.2054312, -2.6400914, 2.6214209
3: -8.5602093, -4.6137319, -8.4776821, -4.6502581, -3.1113091, 3.0996995
4: -12.2624035, -8.2862072, -12.2409687, -8.3911829, -3.8712206, 3.8420839
5: -6.8784237, -3.7693172, -6.8478389, -3.7864528, -2.9433880, 3.0577340
6: -10.9036903, -7.3607702, -10.8407078, -7.4052458, -3.0639853, 3.0264716
7: -3.4293606, -0.3396463, -3.4073963, -0.4115920, -2.9611917, 2.9080534
8: 1.5338030, 3.8494711, 1.5753970, 3.8444228, -2.3000460, 2.2740741
9: -8.6866875, -4.9780312, -8.6495953, -5.0589695, -3.3343573, 3.3318424

Time for backsubstitution: 13.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 4571
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 4682
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 67

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 957

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 565

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6900022, upper bound: 1.7047340
time: 5.61 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6945701, upper bound: 1.7047340
time: 5.71 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -6.0009723, -2.1764812, -5.9640474, -2.1782813, -3.4645929, 3.4419169
1: -6.7709808, -3.7226803, -6.7546673, -3.7647538, -2.7254915, 2.7174041
2: -5.1113548, -2.1590376, -5.1030397, -2.1809533, -2.6570024, 2.6528246
3: -8.5668955, -4.5918565, -8.5185728, -4.6051707, -3.1262159, 3.1404181
4: -12.2802219, -8.2786160, -12.2783337, -8.3626080, -3.9176140, 3.9331188
5: -6.8823743, -3.7586737, -6.8611784, -3.7625577, -3.1020226, 3.0847645
6: -10.9342690, -7.3556490, -10.9053783, -7.3616533, -3.1263442, 3.0637138
7: -3.4392991, -0.3256371, -3.4429109, -0.3820863, -2.9873390, 3.0011396
8: 1.5276604, 3.8648663, 1.5441084, 3.8762331, -2.3187585, 2.3145332
9: -8.6947718, -4.9729424, -8.6770487, -5.0477009, -3.3586845, 3.3884211

Time for backsubstitution: 13.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 4571
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 4682
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 67

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 957

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 565

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7016407, upper bound: 1.7101741
time: 5.24 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7062445, upper bound: 1.7101750
time: 5.63 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -5.9379878, -2.2013087, -5.9936528, -2.1832769, -3.3356524, 3.4405756
1: -6.7105370, -3.8027706, -6.7488761, -3.7267828, -2.7000132, 2.6681457
2: -5.0755606, -2.2253871, -5.1062870, -2.1697125, -2.6074123, 2.6189597
3: -8.4741306, -4.6418257, -8.5602093, -4.6137319, -3.0966740, 3.1321120
4: -12.2319050, -8.4041920, -12.2624035, -8.2862072, -3.8596230, 3.8582115
5: -6.8439574, -3.7901576, -6.8784237, -3.7693172, -3.0541606, 2.9441218
6: -10.8370085, -7.4052439, -10.9036903, -7.3607702, -3.0238447, 3.0248830
7: -3.3861537, -0.4248948, -3.4293606, -0.3396463, -2.9257698, 2.9468656
8: 1.5829763, 3.8261752, 1.5338030, 3.8494711, -2.2664948, 2.2422667
9: -8.6495190, -5.0613446, -8.6866875, -4.9780312, -3.3136768, 3.3313646

Time for backsubstitution: 13.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 5856
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 4682
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 67

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 957

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 565

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6961518, upper bound: 1.6960977
time: 5.12 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7007254, upper bound: 1.6960988
time: 5.03 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -5.9379878, -2.2013087, -6.0127659, -2.1633291, -3.3563070, 3.4542239
1: -6.7105370, -3.8027706, -6.7596622, -3.7057235, -2.7132034, 2.6802211
2: -5.0755606, -2.2253871, -5.1254997, -2.1312397, -2.6282005, 2.6384649
3: -8.4741306, -4.6418257, -8.5754108, -4.6091323, -3.1012211, 3.1473942
4: -12.2319050, -8.4041920, -12.3198128, -8.2614088, -3.8681517, 3.9047771
5: -6.8439574, -3.7901576, -6.8935871, -3.7617893, -3.0613256, 2.9570217
6: -10.8370085, -7.4052439, -10.9157972, -7.3498516, -3.0355220, 3.0369391
7: -3.3861537, -0.4248948, -3.4940827, -0.3177400, -2.9343958, 2.9836102
8: 1.5829763, 3.8261752, 1.5159249, 3.8866234, -2.3036470, 2.2601182
9: -8.6495190, -5.0613446, -8.7179680, -4.9651117, -3.3198009, 3.3622513

Time for backsubstitution: 13.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 5856
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 4682
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 67

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 957

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 565

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6961518, upper bound: 1.7088022
time: 4.92 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7007239, upper bound: 1.7088030
time: 5.54 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -5.9552274, -2.1846676, -6.0009723, -2.1764812, -3.4331474, 3.4629059
1: -6.7588429, -3.7739077, -6.7709808, -3.7226803, -2.7201228, 2.7192392
2: -5.0946350, -2.2009048, -5.1113548, -2.1590376, -2.6471515, 2.6358807
3: -8.5150480, -4.5968103, -8.5668955, -4.5918565, -3.1373634, 3.1356044
4: -12.2690563, -8.3755856, -12.2802219, -8.2786160, -3.9412308, 3.9046364
5: -6.8573217, -3.7663980, -6.8823743, -3.7586737, -3.0812368, 3.1143708
6: -10.9016905, -7.3617578, -10.9342690, -7.3556490, -3.0611258, 3.1320238
7: -3.4215176, -0.3954093, -3.4392991, -0.3256371, -3.0016375, 2.9729958
8: 1.5516357, 3.8578863, 1.5276604, 3.8648663, -2.3048723, 2.3043177
9: -8.6767950, -5.0500617, -8.6947718, -4.9729424, -3.3988314, 3.3557005

Time for backsubstitution: 13.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 5856
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 4682
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 67

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 957

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 565

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7016504, upper bound: 1.7077738
time: 9.16 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7062463, upper bound: 1.7077737
time: 5.48 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -5.9552274, -2.1846676, -6.0201015, -2.1565363, -3.4537058, 3.4765601
1: -6.7588429, -3.7739077, -6.7817388, -3.7016158, -2.7337956, 2.7311990
2: -5.0946350, -2.2009048, -5.1305661, -2.1205626, -2.6679258, 2.6555207
3: -8.5150480, -4.5968103, -8.5820122, -4.5872593, -3.1419115, 3.1508956
4: -12.2690563, -8.3755856, -12.3376360, -8.2538357, -3.9499192, 3.9491754
5: -6.8573217, -3.7663980, -6.8975630, -3.7511303, -3.0884819, 3.1269746
6: -10.9016905, -7.3617578, -10.9463825, -7.3447275, -3.0728102, 3.1441586
7: -3.4215176, -0.3954093, -3.5040226, -0.3037164, -3.0102768, 3.0094154
8: 1.5516357, 3.8578863, 1.5097938, 3.9020138, -2.3200634, 2.3204663
9: -8.6767950, -5.0500617, -8.7260513, -4.9600267, -3.4051189, 3.3865938

Time for backsubstitution: 13.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 5856
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 4682
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 67

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 957

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 565

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7016503, upper bound: 1.7204799
time: 8.56 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7062463, upper bound: 1.7204797
time: 5.45 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -5.9667215, -2.1731696, -5.9535818, -2.1772499, -3.4438453, 3.3571801
1: -6.7565422, -3.7608659, -6.7497339, -3.7684023, -2.7196970, 2.7631335
2: -5.1049585, -2.1704268, -5.0888238, -2.1769028, -2.6515908, 2.6303766
3: -8.5235214, -4.6043611, -8.5213814, -4.6060534, -3.1392260, 3.1316490
4: -12.2928495, -8.3604889, -12.2818031, -8.3800135, -3.9128361, 3.9213142
5: -6.8650274, -3.7615185, -6.8611283, -3.7659416, -3.0651321, 3.0725427
6: -10.9071693, -7.3593745, -10.8989658, -7.3622732, -3.0668836, 3.1093352
7: -3.4559453, -0.3809419, -3.4381523, -0.4057732, -2.9950323, 3.0364597
8: 1.5422993, 3.8831015, 1.5596142, 3.8728576, -2.2815270, 2.2959380
9: -8.6856022, -5.0466866, -8.6792812, -5.0553999, -3.3367338, 3.3372412

Time for backsubstitution: 13.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4571
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 4682
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 67

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4571

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7028770, upper bound: 1.6789828
time: 5.10 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7082859, upper bound: 1.6905772
time: 5.92 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -5.9667215, -2.1731696, -6.0053492, -2.1638446, -3.4578357, 3.4648767
1: -6.7565422, -3.7608659, -6.7738285, -3.7133260, -2.7590966, 2.7371502
2: -5.1049585, -2.1704268, -5.1127052, -2.1326361, -2.6809645, 2.6669738
3: -8.5235214, -4.6043611, -8.5784254, -4.5898585, -3.1450787, 3.1740592
4: -12.2928495, -8.3604889, -12.3169012, -8.2745571, -3.9707909, 3.9564123
5: -6.8650274, -3.7615185, -6.8914137, -3.7560754, -3.0928688, 3.1044607
6: -10.9071693, -7.3593745, -10.9369211, -7.3498440, -3.1295562, 3.1462944
7: -3.4559453, -0.3809419, -3.4726038, -0.3291435, -3.0284257, 3.0348594
8: 1.5422993, 3.8831015, 1.5280190, 3.8829637, -2.3345246, 2.3278215
9: -8.6856022, -5.0466866, -8.7158241, -4.9695349, -3.3986835, 3.3799052

Time for backsubstitution: 13.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4571
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 4682
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 67

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4571

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7028771, upper bound: 1.6892238
time: 5.11 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7082859, upper bound: 1.7008612
time: 5.81 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -6.0112829, -2.1666255, -5.9420724, -2.1886649, -3.4657640, 3.3503547
1: -6.7586179, -3.7099924, -6.7133770, -3.7932198, -2.6979322, 2.7204969
2: -5.1237836, -2.1368575, -5.0769005, -2.1989961, -2.6658220, 2.6376185
3: -8.5738401, -4.6099920, -8.4857836, -4.6398787, -3.1513834, 3.1129799
4: -12.3101015, -8.2626324, -12.2684135, -8.4002647, -3.9098368, 3.9034302
5: -6.8912210, -3.7621872, -6.8529754, -3.7877340, -2.9593940, 3.0640078
6: -10.9144897, -7.3520489, -10.8395681, -7.3995314, -3.0419369, 3.0366201
7: -3.4802256, -0.3183532, -3.4194591, -0.4285722, -2.9915442, 2.9673135
8: 1.5168462, 3.8777370, 1.5833478, 3.8442168, -2.2770133, 2.2913370
9: -8.7139664, -4.9659443, -8.6704683, -5.0580730, -3.3541517, 3.3394942

Time for backsubstitution: 13.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 4571
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 4682
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 67

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 957

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7014083, upper bound: 1.6953780
time: 5.34 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7014083, upper bound: 1.6953766
time: 5.62 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 25.09 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.09
Output dim: 7, lower bound: -1.6797122, upper bound: 1.7047343
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.09
Output dim: 7, lower bound: -1.6843152, upper bound: 1.7047342
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.09
Output dim: 7, lower bound: -1.6912121, upper bound: 1.7101745
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.09
Output dim: 7, lower bound: -1.6959415, upper bound: 1.7101746
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.09
Output dim: 7, lower bound: -1.6900022, upper bound: 1.7047340
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.09
Output dim: 7, lower bound: -1.6945701, upper bound: 1.7047340
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.09
Output dim: 7, lower bound: -1.7016407, upper bound: 1.7101741
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.09
Output dim: 7, lower bound: -1.7062445, upper bound: 1.7101750
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.09
Output dim: 7, lower bound: -1.6961518, upper bound: 1.6960977
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.09
Output dim: 7, lower bound: -1.7007254, upper bound: 1.6960988
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.09
Output dim: 7, lower bound: -1.6961518, upper bound: 1.7088022
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.09
Output dim: 7, lower bound: -1.7007239, upper bound: 1.7088030
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.09
Output dim: 7, lower bound: -1.7016504, upper bound: 1.7077738
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.09
Output dim: 7, lower bound: -1.7062463, upper bound: 1.7077737
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.09
Output dim: 7, lower bound: -1.7016503, upper bound: 1.7204799
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.09
Output dim: 7, lower bound: -1.7062463, upper bound: 1.7204797
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.09
Output dim: 7, lower bound: -1.7028770, upper bound: 1.6789828
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.09
Output dim: 7, lower bound: -1.7082859, upper bound: 1.6905772
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.09
Output dim: 7, lower bound: -1.7028771, upper bound: 1.6892238
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.09
Output dim: 7, lower bound: -1.7082859, upper bound: 1.7008612
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.09
Output dim: 7, lower bound: -1.7014083, upper bound: 1.6953780
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.09
Output dim: 7, lower bound: -1.7014083, upper bound: 1.6953766
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.09
Output dim: 7, lower bound: -1.7185957, upper bound: 1.7008628
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 25.09
Output dim: 7, lower bound: -1.7083008, upper bound: 1.7083009
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 25.09
Output dim: 7, lower bound: -1.7083008, upper bound: 1.7186104
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 25.09
Output dim: 7, lower bound: -1.7069439, upper bound: 1.7130983
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.09
Output dim: 7, lower bound: -1.7185957, upper bound: 1.7185986
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=3.0680713653564453
rel_dist={7: [-1.7205280069910254, 1.7205289191635718]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5871
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 5856
type: A, layer: 1, pos: 4571
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 4682
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 67

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5871

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6088340, upper bound: 1.6196704
time: 5.06 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6196703, upper bound: 1.6196701
time: 5.38 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 10.61 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 10.61
Output dim: 7, lower bound: -1.6088340, upper bound: 1.6196704
IS_A2, status: Status.UNKNOWN, split count: 1, time: 10.61
Output dim: 7, lower bound: -1.6196703, upper bound: 1.6196701

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -5.9552526, -2.1846352, -5.9692736, -2.1747236, -3.3816328, 3.3855667
1: -6.7589445, -3.7738898, -6.7663779, -3.7623272, -2.6616516, 2.6634641
2: -5.0946593, -2.2008746, -5.1095161, -2.1817644, -2.5940442, 2.5889993
3: -8.5150728, -4.5967035, -8.5228939, -4.5939951, -3.0495548, 3.0542817
4: -12.2691593, -8.3755531, -12.2976894, -8.3548145, -3.8748751, 3.8829083
5: -6.8573399, -3.7663674, -6.8653593, -3.7606406, -3.0265627, 3.0300231
6: -10.9017916, -7.3617334, -10.9102459, -7.3561149, -3.0493674, 3.0533359
7: -3.4215825, -0.3953617, -3.4544997, -0.3756020, -2.9616899, 2.9675257
8: 1.5516086, 3.8579321, 1.5370669, 3.8763895, -2.2706661, 2.2672439
9: -8.6768417, -5.0500345, -8.6932716, -5.0393834, -3.2900505, 3.2950597

Time for backsubstitution: 13.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 5856
type: B, layer: 1, pos: 4571
type: B, layer: 1, pos: 5871
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 4572
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 4682
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 67

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 957

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5856

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.6085919, upper bound: 1.6114452
time: 5.07 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6088165, upper bound: 1.6196520
time: 5.74 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -5.9742651, -2.1646063, -5.9743257, -2.1644878, -3.4115753, 3.3975577
1: -6.7698288, -3.7524176, -6.7698898, -3.7523305, -2.6958609, 2.6881580
2: -5.1138420, -2.1623573, -5.1139402, -2.1622076, -2.6347432, 2.6292586
3: -8.5302773, -4.5920773, -8.5304079, -4.5920668, -3.0642548, 3.0668020
4: -12.3267860, -8.3508511, -12.3269901, -8.3508511, -3.9260416, 3.9356880
5: -6.8726435, -3.7589469, -6.8727093, -3.7589006, -3.0432949, 3.0414028
6: -10.9138498, -7.3507442, -10.9138498, -7.3506889, -3.0703182, 3.0687652
7: -3.4867694, -0.3735499, -3.4868200, -0.3735502, -3.0256886, 3.0242383
8: 1.5337977, 3.8952694, 1.5337811, 3.8952727, -2.3082948, 2.2876923
9: -8.7084379, -5.0371690, -8.7087078, -5.0371380, -3.3155165, 3.3237829

Time for backsubstitution: 13.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 5856
type: B, layer: 1, pos: 5871
type: B, layer: 1, pos: 4571
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 4572
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 4682
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 67

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 957

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.6179064, upper bound: 1.6043999
time: 5.13 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.6179065, upper bound: 1.6179066
time: 5.55 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 24.73 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 24.73
Output dim: 7, lower bound: -1.6085919, upper bound: 1.6114452
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 24.73
Output dim: 7, lower bound: -1.6088165, upper bound: 1.6196520
IS_A2_B1, status: Status.VERIFIED, split count: 2, time: 24.73
Output dim: 7, lower bound: -1.6179064, upper bound: 1.6043999
IS_A2_B2, status: Status.VERIFIED, split count: 2, time: 24.73
Output dim: 7, lower bound: -1.6179065, upper bound: 1.6179066

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -5.9552393, -2.1846476, -6.0150766, -2.1666217, -3.3901138, 3.4223788
1: -6.7589087, -3.7738981, -6.7783871, -3.7112901, -2.7010093, 2.6759090
2: -5.0946460, -2.2008791, -5.1262255, -2.1399295, -2.6191578, 2.6083660
3: -8.5150604, -4.5967407, -8.5747414, -4.5891204, -3.0528097, 3.0941939
4: -12.2691021, -8.3755713, -12.3086262, -8.2578192, -3.9172354, 3.8915305
5: -6.8573284, -3.7663693, -6.8904567, -3.7528608, -3.0307097, 3.0677638
6: -10.9017801, -7.3617487, -10.9428043, -7.3500671, -3.0540504, 3.0907378
7: -3.4215331, -0.3953772, -3.4719272, -0.3057876, -2.9945817, 2.9812431
8: 1.5516253, 3.8579254, 1.5130792, 3.8832116, -2.2790248, 2.2925439
9: -8.6768084, -5.0500536, -8.7109842, -4.9622450, -3.3446035, 3.3078852

Time for backsubstitution: 13.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 4571
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 5856
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 4682
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 67

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 957

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4571

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.6029903, upper bound: 1.6091640
time: 4.97 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6088028, upper bound: 1.6196373
time: 5.30 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 30.12 seconds
IS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 30.12
Output dim: 7, lower bound: -1.6029903, upper bound: 1.6091640
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 30.12
Output dim: 7, lower bound: -1.6088028, upper bound: 1.6196373

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -5.9552259, -2.1846700, -6.0150704, -2.1666336, -3.3900776, 3.4224463
1: -6.7588363, -3.7739077, -6.7783451, -3.7112958, -2.6726503, 2.6758595
2: -5.0946321, -2.2009053, -5.1262178, -2.1399450, -2.6143742, 2.6060557
3: -8.5150461, -4.5968165, -8.5747328, -4.5891643, -3.0527449, 3.0542011
4: -12.2690468, -8.3755875, -12.3085938, -8.2578268, -3.8977690, 3.8800073
5: -6.8573198, -3.7663984, -6.8904514, -3.7528777, -3.0315089, 3.0674591
6: -10.9016895, -7.3617601, -10.9427481, -7.3500748, -2.9947100, 3.0711744
7: -3.4215112, -0.3954115, -3.4719133, -0.3058093, -2.9841404, 2.9599698
8: 1.5516386, 3.8578854, 1.5130877, 3.8831882, -2.2708602, 2.2743444
9: -8.6767893, -5.0500650, -8.7109737, -4.9622521, -3.3445816, 3.3084431

Time for backsubstitution: 13.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 5871
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 4572
type: B, layer: 1, pos: 4571
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 4682
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 67

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 957

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 565

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.6087969, upper bound: 1.6159067
time: 5.17 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6087969, upper bound: 1.6196310
time: 5.30 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 30.39 seconds
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 30.39
Output dim: 7, lower bound: -1.6087969, upper bound: 1.6159067
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.39
Output dim: 7, lower bound: -1.6087969, upper bound: 1.6196310

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -5.9552250, -2.1846771, -6.0150681, -2.1666574, -3.3224449, 3.3982346
1: -6.7588329, -3.7739089, -6.7783303, -3.7112963, -2.6541400, 2.6309896
2: -5.0946326, -2.2009063, -5.1262164, -2.1399488, -2.6026554, 2.6036663
3: -8.5150461, -4.5968184, -8.5747299, -4.5891714, -3.0311985, 3.0433090
4: -12.2690420, -8.3755875, -12.3085766, -8.2578287, -3.8775783, 3.8429205
5: -6.8573189, -3.7663989, -6.8904486, -3.7528796, -3.0234108, 3.0807276
6: -10.9016905, -7.3617620, -10.9427462, -7.3500838, -2.9795971, 3.0584350
7: -3.4215074, -0.3954115, -3.4719040, -0.3058100, -2.9660835, 2.9416218
8: 1.5516386, 3.8578806, 1.5130892, 3.8831706, -2.1779175, 2.2552328
9: -8.6767883, -5.0500650, -8.7109737, -4.9622507, -3.3328424, 3.3064919

Time for backsubstitution: 13.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 5856
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 4682
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 67

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 957

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6220

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6065662, upper bound: 1.6196265
time: 5.10 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6087923, upper bound: 1.6196269
time: 5.26 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 30.35 seconds
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 30.35
Output dim: 7, lower bound: -1.6065662, upper bound: 1.6196265
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 30.35
Output dim: 7, lower bound: -1.6087923, upper bound: 1.6196269

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -5.9481306, -2.1881952, -6.0112519, -2.1686871, -3.3015127, 3.3805907
1: -6.7538199, -3.7888720, -6.7754812, -3.7198365, -2.6321030, 2.6024785
2: -5.0840125, -2.2049856, -5.1201844, -2.1422062, -2.5875039, 2.5913315
3: -8.5092869, -4.5989513, -8.5716534, -4.5904007, -3.0224237, 3.0356748
4: -12.2655907, -8.3863554, -12.3065338, -8.2633553, -3.8635206, 3.8263259
5: -6.8494253, -3.7756128, -6.8861542, -3.7582493, -3.0082235, 3.0632744
6: -10.8972960, -7.3678188, -10.9404640, -7.3534083, -2.9656577, 3.0459476
7: -3.4098492, -0.4014678, -3.4654076, -0.3088627, -2.9470515, 2.9248946
8: 1.5695128, 3.8552928, 1.5232925, 3.8817372, -2.1478539, 2.2314439
9: -8.6535206, -5.0561275, -8.6976442, -4.9650698, -3.3086009, 3.2892447

Time for backsubstitution: 14.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 5871
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 4572
type: B, layer: 1, pos: 4571
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 4682
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 67

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 957

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5871

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.6065662, upper bound: 1.6112442
time: 5.22 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6065662, upper bound: 1.6196265
time: 5.24 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -5.9832196, -2.1652570, -6.0150623, -2.1666596, -3.3493948, 3.4147158
1: -6.8074212, -3.7583718, -6.7783270, -3.7113080, -2.6771045, 2.6400688
2: -5.1052408, -2.1671493, -5.1262059, -2.1399515, -2.6174645, 2.6328564
3: -8.5326452, -4.5851226, -8.5747261, -4.5891728, -3.0488815, 3.0544665
4: -12.2996302, -8.3639994, -12.3085747, -8.2578354, -3.8958826, 3.8505867
5: -6.8907347, -3.7560887, -6.8904428, -3.7528901, -3.0570297, 3.0887654
6: -10.9161530, -7.3321085, -10.9427443, -7.3500857, -3.0098467, 3.0782130
7: -3.4411705, -0.3662403, -3.4718916, -0.3058157, -2.9847040, 2.9535179
8: 1.5310683, 3.8863258, 1.5131035, 3.8831687, -2.1891425, 2.2604666
9: -8.7018948, -5.0108237, -8.7109547, -4.9622569, -3.3531790, 3.3358259

Time for backsubstitution: 13.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 5871
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 4572
type: B, layer: 1, pos: 4571
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 4682
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 67

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 957

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5871

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.6087923, upper bound: 1.6112437
time: 4.99 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6087923, upper bound: 1.6196265
time: 5.57 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 30.51 seconds
IS_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 30.51
Output dim: 7, lower bound: -1.6065662, upper bound: 1.6112442
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 30.51
Output dim: 7, lower bound: -1.6065662, upper bound: 1.6196265
IS_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 30.51
Output dim: 7, lower bound: -1.6087923, upper bound: 1.6112437
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 30.51
Output dim: 7, lower bound: -1.6087923, upper bound: 1.6196265

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -5.9481306, -2.1881952, -6.0162830, -2.1585872, -3.3117113, 3.3798091
1: -6.7538199, -3.7888720, -6.7788563, -3.7101560, -2.6366820, 2.6041243
2: -5.0840125, -2.2049856, -5.1245303, -2.1228259, -2.5945096, 2.5909486
3: -8.5092869, -4.5989513, -8.5789385, -4.5885129, -3.0241370, 3.0416584
4: -12.2655907, -8.3863554, -12.3355770, -8.2593670, -3.8518305, 3.8301630
5: -6.8494253, -3.7756128, -6.8932500, -3.7565055, -3.0080814, 3.0667686
6: -10.8972960, -7.3678188, -10.9440899, -7.3480644, -2.9714370, 3.0480330
7: -3.4098492, -0.4014678, -3.4975142, -0.3067520, -2.9344988, 2.9315116
8: 1.5695128, 3.8552928, 1.5199990, 3.9005580, -2.1493475, 2.2253108
9: -8.6535206, -5.0561275, -8.7127247, -4.9628468, -3.3038759, 3.3043318

Time for backsubstitution: 13.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 5856
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 4682
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 67

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 957

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5856

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.5983655, upper bound: 1.6194049
time: 5.10 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.5983655, upper bound: 1.6114181
time: 5.60 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -5.9832196, -2.1652570, -6.0200925, -2.1565633, -3.3519926, 3.4139383
1: -6.8074212, -3.7583718, -6.7817121, -3.7016275, -2.6816864, 2.6398699
2: -5.1052408, -2.1671493, -5.1305528, -2.1205707, -2.6244726, 2.6296618
3: -8.5326452, -4.5851226, -8.5820036, -4.5872750, -3.0488615, 3.0604303
4: -12.2996302, -8.3639994, -12.3376122, -8.2538481, -3.8841972, 3.8544359
5: -6.8907347, -3.7560887, -6.8975534, -3.7511449, -3.0548801, 3.0922699
6: -10.9161530, -7.3321085, -10.9463682, -7.3447428, -3.0156260, 3.0803008
7: -3.4411705, -0.3662403, -3.5039992, -0.3037243, -2.9721689, 2.9601469
8: 1.5310683, 3.8863258, 1.5098095, 3.9019890, -2.1906400, 2.2543430
9: -8.7018948, -5.0108237, -8.7260284, -4.9600325, -3.3484573, 3.3390555

Time for backsubstitution: 13.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 5856
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 4682
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 67

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 957

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.6077163, upper bound: 1.6160757
time: 5.34 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6087863, upper bound: 1.6196201
time: 5.03 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 30.38 seconds
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 30.38
Output dim: 7, lower bound: -1.5983655, upper bound: 1.6194049
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 30.38
Output dim: 7, lower bound: -1.5983655, upper bound: 1.6114181
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 30.38
Output dim: 7, lower bound: -1.6077163, upper bound: 1.6160757
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 30.38
Output dim: 7, lower bound: -1.6087863, upper bound: 1.6196201

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -5.9421225, -2.1934624, -6.0157127, -2.1586487, -3.2560720, 3.3740277
1: -6.7419319, -3.7929845, -6.7787409, -3.7108018, -2.6670666, 2.5973322
2: -5.0768428, -2.2074380, -5.1243510, -2.1233106, -2.5785327, 2.5869250
3: -8.5037155, -4.6101732, -8.5780830, -4.5885587, -3.0163226, 3.0300064
4: -12.2417660, -8.3944550, -12.3354788, -8.2607670, -3.8389559, 3.8155069
5: -6.8439493, -3.7775784, -6.8927436, -3.7565608, -2.9996252, 3.0358634
6: -10.8917246, -7.3740706, -10.9435101, -7.3481178, -2.9628720, 2.9892907
7: -3.3930905, -0.4084141, -3.4972880, -0.3077903, -2.9480772, 2.9195056
8: 1.5771785, 3.8521838, 1.5202560, 3.9004884, -2.1424150, 2.1787784
9: -8.6349907, -5.0649195, -8.7125416, -4.9639053, -3.2651997, 3.2954063

Time for backsubstitution: 13.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 4572
type: B, layer: 1, pos: 4571
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 4682
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 67

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 957

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4626

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.5951484, upper bound: 1.6183156
time: 5.64 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.5983592, upper bound: 1.6193978
time: 5.40 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -5.9831777, -2.1652625, -6.0200691, -2.1565671, -3.3443542, 3.4081435
1: -6.8074121, -3.7583880, -6.7817068, -3.7016373, -2.6699448, 2.6359308
2: -5.1052270, -2.1671698, -5.1305432, -2.1205835, -2.6210423, 2.6241961
3: -8.5326118, -4.5851269, -8.5819864, -4.5872803, -3.0299654, 3.0462151
4: -12.2995853, -8.3640099, -12.3375874, -8.2538548, -3.8716831, 3.8387735
5: -6.8907089, -3.7560983, -6.8975396, -3.7511511, -3.0481958, 3.0833106
6: -10.9161253, -7.3321176, -10.9463511, -7.3447466, -3.0091186, 3.0703959
7: -3.4411592, -0.3662586, -3.5039916, -0.3037353, -2.9654760, 2.9546440
8: 1.5310917, 3.8863230, 1.5098233, 3.9019866, -2.1775339, 2.2448273
9: -8.7018404, -5.0108342, -8.7259970, -4.9600391, -3.3280816, 3.3206739

Time for backsubstitution: 13.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 4572
type: B, layer: 1, pos: 4571
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 4682
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 67

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 957

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4572

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6051169, upper bound: 1.6189945
time: 5.47 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6087804, upper bound: 1.6196144
time: 17.29 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 42.68 seconds
IS_A1_B2_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 42.68
Output dim: 7, lower bound: -1.5951484, upper bound: 1.6183156
IS_A1_B2_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 42.68
Output dim: 7, lower bound: -1.5983592, upper bound: 1.6193978
IS_A1_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 42.68
Output dim: 7, lower bound: -1.6051169, upper bound: 1.6189945
IS_A1_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 42.68
Output dim: 7, lower bound: -1.6087804, upper bound: 1.6196144

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -5.9420986, -2.1934657, -6.0156689, -2.1586537, -3.2509890, 3.3663738
1: -6.7419281, -3.7929945, -6.7787328, -3.7108185, -2.6627851, 2.5855899
2: -5.0768337, -2.2074482, -5.1243358, -2.1233292, -2.5730486, 2.5834739
3: -8.5036983, -4.6101761, -8.5780516, -4.5885611, -3.0020447, 3.0112777
4: -12.2417402, -8.3944635, -12.3354359, -8.2607794, -3.8214040, 3.8030066
5: -6.8439350, -3.7775843, -6.8927178, -3.7565708, -3.0002561, 3.0320458
6: -10.8917103, -7.3740749, -10.9434814, -7.3481283, -2.9628506, 2.9795160
7: -3.3930860, -0.4084244, -3.4972782, -0.3078079, -2.9417887, 2.9127860
8: 1.5771928, 3.8521829, 1.5202794, 3.9004865, -2.1329193, 2.1659095
9: -8.6349602, -5.0649261, -8.7124882, -4.9639187, -3.2466779, 3.2770290

Time for backsubstitution: 13.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 4682
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 67

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 957

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 565

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.5946235, upper bound: 1.6193979
time: 5.72 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.5946235, upper bound: 1.6156676
time: 5.94 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -5.9774628, -2.1952410, -5.9860115, -2.2141280, -3.2757740, 3.3225770
1: -6.7831392, -3.7674518, -6.7341080, -3.7359869, -2.6028495, 2.5828862
2: -5.0998302, -2.1765113, -5.1156030, -2.1403799, -2.5385919, 2.5790176
3: -8.5224476, -4.6249766, -8.5352993, -4.6615024, -2.9390459, 2.9405863
4: -12.2525005, -8.3728991, -12.2502766, -8.3023529, -3.7527299, 3.7613783
5: -6.8817058, -3.7754004, -6.8681259, -3.7901316, -2.9986625, 2.9591591
6: -10.8686609, -7.3444543, -10.8570900, -7.3956594, -2.8905373, 2.9678144
7: -3.4318168, -0.3789835, -3.4725378, -0.3287182, -2.9271259, 2.8985479
8: 1.5395398, 3.8639474, 1.5379634, 3.8608756, -2.0399575, 2.1739538
9: -8.6748257, -5.0181303, -8.6642513, -4.9829502, -3.2608399, 3.2485147

Time for backsubstitution: 13.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 5856
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 4682
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 67

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 957

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5856

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.5970032, upper bound: 1.6188210
time: 5.17 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.5970032, upper bound: 1.6108912
time: 5.89 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -5.9831753, -2.1652818, -6.0200620, -2.1566169, -3.3321409, 3.3954756
1: -6.8074031, -3.7583938, -6.7816877, -3.7016478, -2.6659818, 2.6320987
2: -5.1052246, -2.1671729, -5.1305361, -2.1205916, -2.6210256, 2.6258860
3: -8.5326080, -4.5851398, -8.5819740, -4.5873194, -3.0238619, 3.0396688
4: -12.2995567, -8.3640146, -12.3375206, -8.2538614, -3.8572145, 3.8319886
5: -6.8907056, -3.7561038, -6.8975286, -3.7511649, -3.0451441, 3.0810328
6: -10.9161062, -7.3321233, -10.9463024, -7.3447614, -2.9949145, 3.0618727
7: -3.4411561, -0.3662634, -3.5039809, -0.3037510, -2.9632869, 2.9514596
8: 1.5310946, 3.8863106, 1.5098329, 3.9019589, -2.1754582, 2.2409337
9: -8.7018280, -5.0108390, -8.7259712, -4.9600496, -3.3263912, 3.3187461

Time for backsubstitution: 13.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 5856
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 4682
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 67

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 957

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5856

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6005654, upper bound: 1.6193918
time: 5.11 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.6005654, upper bound: 1.6119773
time: 5.49 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 30.47 seconds
IS_A1_B2_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 30.47
Output dim: 7, lower bound: -1.5946235, upper bound: 1.6193979
IS_A1_B2_A2_B2_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 9, time: 30.47
Output dim: 7, lower bound: -1.5946235, upper bound: 1.6156676
IS_A1_B2_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 30.47
Output dim: 7, lower bound: -1.5970032, upper bound: 1.6188210
IS_A1_B2_A2_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 30.47
Output dim: 7, lower bound: -1.5970032, upper bound: 1.6108912
IS_A1_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 30.47
Output dim: 7, lower bound: -1.6005654, upper bound: 1.6193918
IS_A1_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 30.47
Output dim: 7, lower bound: -1.6005654, upper bound: 1.6119773

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -5.8898582, -2.2898755, -6.0144558, -2.1588383, -3.1733322, 3.2676580
1: -6.6748333, -3.8235137, -6.7784886, -3.7122161, -2.6043739, 2.5629523
2: -5.0467811, -2.2383156, -5.1239471, -2.1243646, -2.5463586, 2.5534704
3: -8.4670372, -4.6438799, -8.5762300, -4.5886707, -2.9662623, 2.9757414
4: -12.1726093, -8.4335003, -12.3352308, -8.2637596, -3.7539349, 3.7697482
5: -6.8110147, -3.8049240, -6.8916378, -3.7566919, -2.9643784, 2.9826374
6: -10.8620224, -7.4386578, -10.9422464, -7.3482890, -2.9262843, 2.9131026
7: -3.3344471, -0.4371216, -3.4967725, -0.3099968, -2.8801427, 2.8824942
8: 1.6115274, 3.7447233, 1.5208292, 3.9002151, -2.1139328, 2.0550511
9: -8.5894718, -5.0784087, -8.7120523, -4.9661722, -3.2047358, 3.2580843

Time for backsubstitution: 13.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 4572
type: B, layer: 1, pos: 4571
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 4682
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 67

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 957

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4572

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.5911810, upper bound: 1.6188215
time: 5.37 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.5946176, upper bound: 1.6193911
time: 5.33 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -5.9715500, -2.2004762, -5.9860115, -2.2141280, -3.2680206, 3.3172574
1: -6.7709441, -3.7718191, -6.7341080, -3.7359869, -2.5897727, 2.5762258
2: -5.0926971, -2.1789052, -5.1156030, -2.1403799, -2.5281615, 2.5752177
3: -8.5173397, -4.6363516, -8.5352993, -4.6615024, -2.9288440, 2.9265370
4: -12.2283773, -8.3821297, -12.2502766, -8.3023529, -3.7285848, 3.7460976
5: -6.8765726, -3.7775178, -6.8681259, -3.7901316, -2.9904284, 2.9531634
6: -10.8635092, -7.3509026, -10.8570900, -7.3956594, -2.8810835, 2.9585092
7: -3.4149642, -0.3855464, -3.4725378, -0.3287182, -2.9060326, 2.8864298
8: 1.5470085, 3.8606973, 1.5379634, 3.8608756, -2.0332017, 2.1709850
9: -8.6560936, -5.0262861, -8.6642513, -4.9829502, -3.2404723, 3.2340379

Time for backsubstitution: 13.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 4571
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 4682
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 67

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 957

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4571

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.5864989, upper bound: 1.6129016
time: 5.04 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.5864988, upper bound: 1.6188188
time: 5.09 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -5.9772587, -2.1705174, -6.0200620, -2.1566169, -3.3243847, 3.3901541
1: -6.7951875, -3.7627656, -6.7816877, -3.7016478, -2.6528730, 2.6253381
2: -5.0980968, -2.1695738, -5.1305361, -2.1205916, -2.6106658, 2.6220894
3: -8.5274982, -4.5965300, -8.5819740, -4.5873194, -3.0141153, 3.0256107
4: -12.2754097, -8.3732471, -12.3375206, -8.2538614, -3.8330455, 3.8170791
5: -6.8855743, -3.7582197, -6.8975286, -3.7511649, -3.0369158, 3.0738113
6: -10.9109516, -7.3385706, -10.9463024, -7.3447614, -2.9854617, 3.0524797
7: -3.4242857, -0.3728266, -3.5039809, -0.3037510, -2.9421821, 2.9393964
8: 1.5385270, 3.8830552, 1.5098329, 3.9019589, -2.1686537, 2.2379570
9: -8.6831045, -5.0189986, -8.7259712, -4.9600496, -3.3057423, 3.3042889

Time for backsubstitution: 13.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 4571
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 4682
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 67

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 957

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4571

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.5902517, upper bound: 1.6136101
time: 5.46 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.5902517, upper bound: 1.6193953
time: 5.81 seconds

## Summary of splitting at layer (split count: 9)
- Time for IS candidates: 31.41 seconds
IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 31.41
Output dim: 7, lower bound: -1.5911810, upper bound: 1.6188215
IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 31.41
Output dim: 7, lower bound: -1.5946176, upper bound: 1.6193911
IS_A1_B2_A2_B2_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 10, time: 31.41
Output dim: 7, lower bound: -1.5864989, upper bound: 1.6129016
IS_A1_B2_A2_B2_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 31.41
Output dim: 7, lower bound: -1.5864988, upper bound: 1.6188188
IS_A1_B2_A2_B2_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 10, time: 31.41
Output dim: 7, lower bound: -1.5902517, upper bound: 1.6136101
IS_A1_B2_A2_B2_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 31.41
Output dim: 7, lower bound: -1.5902517, upper bound: 1.6193953

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -5.8843765, -2.3198600, -5.9804182, -2.2164066, -3.1056099, 3.1820164
1: -6.6507773, -3.8325443, -6.7309494, -3.7465472, -2.5454803, 2.5098851
2: -5.0414224, -2.2475011, -5.1090021, -2.1440232, -2.4607301, 2.4739947
3: -8.4569521, -4.6836390, -8.5295639, -4.6627522, -2.8701558, 2.8574100
4: -12.1263733, -8.4423418, -12.2483015, -8.3122530, -3.6701088, 3.6838598
5: -6.8020024, -3.8241818, -6.8622956, -3.7956777, -2.8615212, 2.8804314
6: -10.8145781, -7.4508677, -10.8530006, -7.3991804, -2.8064442, 2.8104715
7: -3.3254480, -0.4498253, -3.4654059, -0.3349738, -2.8438158, 2.8262787
8: 1.6201196, 3.7223639, 1.5489883, 3.8592443, -1.9763803, 1.9428594
9: -8.5624409, -5.0856900, -8.6502600, -4.9890280, -3.1346111, 3.1834407

Time for backsubstitution: 13.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 4682
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 67

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 957

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5858

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.5672967, upper bound: 1.5853738
time: 5.27 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.5911804, upper bound: 1.6188184
time: 5.48 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -5.8898554, -2.2898953, -6.0144496, -2.1588888, -3.1606026, 3.2549927
1: -6.6748257, -3.8235168, -6.7784691, -3.7122254, -2.6009483, 2.5591280
2: -5.0467782, -2.2383199, -5.1239405, -2.1243734, -2.5463409, 2.5551641
3: -8.4670315, -4.6438942, -8.5762205, -4.5887094, -2.9601765, 2.9695282
4: -12.1725817, -8.4335060, -12.3351669, -8.2637691, -3.7401190, 3.7629855
5: -6.8110099, -3.8049312, -6.8916283, -3.7567060, -2.9639087, 2.9835300
6: -10.8620024, -7.4386621, -10.9421978, -7.3483043, -2.9120841, 2.9042437
7: -3.3344436, -0.4371290, -3.4967606, -0.3100119, -2.8780890, 2.8793273
8: 1.6115308, 3.7447119, 1.5208364, 3.9001870, -2.1118407, 2.0512345
9: -8.5894613, -5.0784111, -8.7120275, -4.9661779, -3.2036858, 3.2561469

Time for backsubstitution: 13.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 4682
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 67

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 957

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5858

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.5704487, upper bound: 1.5854513
time: 5.69 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.5946171, upper bound: 1.6193924
time: 5.80 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -5.9715500, -2.2004762, -5.9860048, -2.2141376, -3.2681007, 3.3172495
1: -6.7709441, -3.7718191, -6.7340789, -3.7359912, -2.5885119, 2.5629547
2: -5.0926971, -2.1789052, -5.1155977, -2.1403913, -2.5281582, 2.5740201
3: -8.5173397, -4.6363516, -8.5352964, -4.6615338, -2.9092841, 2.9263859
4: -12.2283773, -8.3821297, -12.2502537, -8.3023605, -3.7285776, 3.7391839
5: -6.8765726, -3.7775178, -6.8681240, -3.7901435, -2.9904242, 2.9531591
6: -10.8635092, -7.3509026, -10.8570557, -7.3956652, -2.8809090, 2.9264772
7: -3.4149642, -0.3855464, -3.4725275, -0.3287325, -2.8955350, 2.8864179
8: 1.5470085, 3.8606973, 1.5379705, 3.8608594, -2.0248511, 2.1688566
9: -8.6560936, -5.0262861, -8.6642437, -4.9829540, -3.2408628, 3.2340283

Time for backsubstitution: 13.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 4682
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 67

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 957

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 565

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.5828665, upper bound: 1.6188204
time: 5.14 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.5828665, upper bound: 1.6151806
time: 5.34 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -5.9772587, -2.1705174, -6.0200567, -2.1566248, -3.3244677, 3.3901482
1: -6.7951875, -3.7627656, -6.7816572, -3.7016518, -2.6515903, 2.6115029
2: -5.0980968, -2.1695738, -5.1305313, -2.1206036, -2.6094236, 2.6209064
3: -8.5274982, -4.5965300, -8.5819683, -4.5873528, -2.9926267, 3.0254498
4: -12.2754097, -8.3732471, -12.3374987, -8.2538700, -3.8330393, 3.8131485
5: -6.8855743, -3.7582197, -6.8975248, -3.7511775, -3.0369110, 3.0743895
6: -10.9109516, -7.3385706, -10.9462662, -7.3447661, -2.9852777, 3.0202100
7: -3.4242857, -0.3728266, -3.5039716, -0.3037658, -2.9316740, 2.9393859
8: 1.5385270, 3.8830552, 1.5098376, 3.9019389, -2.1587753, 2.2358449
9: -8.6831045, -5.0189986, -8.7259636, -4.9600534, -3.3063288, 3.3042822

Time for backsubstitution: 13.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 4682
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 67

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 957

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 565

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.5865395, upper bound: 1.6193948
time: 6.32 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.5865396, upper bound: 1.6156636
time: 5.39 seconds

## Summary of splitting at layer (split count: 10)
- Time for IS candidates: 31.73 seconds
IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 11, time: 31.73
Output dim: 7, lower bound: -1.5672967, upper bound: 1.5853738
IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 11, time: 31.73
Output dim: 7, lower bound: -1.5911804, upper bound: 1.6188184
IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 11, time: 31.73
Output dim: 7, lower bound: -1.5704487, upper bound: 1.5854513
IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 11, time: 31.73
Output dim: 7, lower bound: -1.5946171, upper bound: 1.6193924
IS_A1_B2_A2_B2_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 11, time: 31.73
Output dim: 7, lower bound: -1.5828665, upper bound: 1.6188204
IS_A1_B2_A2_B2_A2_B2_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 11, time: 31.73
Output dim: 7, lower bound: -1.5828665, upper bound: 1.6151806
IS_A1_B2_A2_B2_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 11, time: 31.73
Output dim: 7, lower bound: -1.5865395, upper bound: 1.6193948
IS_A1_B2_A2_B2_A2_B2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 11, time: 31.73
Output dim: 7, lower bound: -1.5865396, upper bound: 1.6156636

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -5.8843741, -2.3198690, -5.9802656, -2.2164257, -3.1032362, 3.1769037
1: -6.6507759, -3.8325467, -6.7309189, -3.7467172, -2.5334535, 2.4981418
2: -5.0414176, -2.2475028, -5.1089535, -2.1441550, -2.4536366, 2.4589760
3: -8.4569502, -4.6836529, -8.5293350, -4.6627684, -2.8481688, 2.8010087
4: -12.1263733, -8.4423428, -12.2482738, -8.3126268, -3.6617475, 3.6803746
5: -6.8019996, -3.8241868, -6.8621616, -3.7956951, -2.8426557, 2.8558960
6: -10.8145790, -7.4508781, -10.8528461, -7.3992000, -2.7943239, 2.7963612
7: -3.3254433, -0.4498274, -3.4653482, -0.3352404, -2.8319368, 2.8152401
8: 1.6201205, 3.7223501, 1.5490570, 3.8592205, -1.9639025, 1.9164317
9: -8.5624323, -5.0856910, -8.6502085, -4.9893093, -3.1178012, 3.1697211

Time for backsubstitution: 13.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 4571
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 4682
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 67

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 957

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4571

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.5806746, upper bound: 1.6129009
time: 5.47 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.5806746, upper bound: 1.6188198
time: 6.33 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -5.8898511, -2.2899048, -6.0142965, -2.1589069, -3.1584721, 3.2498689
1: -6.6748257, -3.8235188, -6.7784376, -3.7123973, -2.5893197, 2.5473554
2: -5.0467753, -2.2383206, -5.1238918, -2.1245041, -2.5380230, 2.5439963
3: -8.4670305, -4.6439099, -8.5759916, -4.5887246, -2.9379444, 2.9088039
4: -12.1725779, -8.4335079, -12.3351364, -8.2641411, -3.7311206, 3.7531674
5: -6.8110065, -3.8049362, -6.8914914, -3.7567239, -2.9456444, 2.9655046
6: -10.8619986, -7.4386721, -10.9420433, -7.3483219, -2.8999615, 2.8906231
7: -3.3344407, -0.4371309, -3.4966977, -0.3102770, -2.8660631, 2.8681521
8: 1.6115322, 3.7446985, 1.5209069, 3.9001646, -2.0988350, 2.0241811
9: -8.5894527, -5.0784121, -8.7119741, -4.9664612, -3.1867032, 3.2424555

Time for backsubstitution: 13.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 4571
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 4682
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 67

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 957

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4571

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.5843205, upper bound: 1.6136063
time: 5.63 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.5843206, upper bound: 1.6193927
time: 5.93 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -5.9169521, -2.2968712, -5.9857817, -2.2141612, -3.2094116, 3.2192945
1: -6.7016807, -3.8026071, -6.7340345, -3.7362428, -2.5278859, 2.5399539
2: -5.0615234, -2.2095766, -5.1155267, -2.1405809, -2.5019231, 2.5444407
3: -8.4805002, -4.6711044, -8.5349607, -4.6615510, -2.8708363, 2.8903642
4: -12.1567879, -8.4216137, -12.2502174, -8.3028927, -3.6579938, 3.6991363
5: -6.8434973, -3.8062851, -6.8679247, -3.7901666, -2.9564829, 2.9002898
6: -10.8336887, -7.4168768, -10.8568249, -7.3956866, -2.8347759, 2.8598223
7: -3.3545589, -0.4146495, -3.4724450, -0.3291247, -2.8336391, 2.8555367
8: 1.5814261, 3.7517676, 1.5380683, 3.8608308, -2.0069044, 2.0576835
9: -8.6079903, -5.0401077, -8.6641722, -4.9833708, -3.1965618, 3.2149940

Time for backsubstitution: 13.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 4682
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 67

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 957

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4573

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.5883981, upper bound: 1.6188064
time: 5.36 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.5902911, upper bound: 1.6188148
time: 6.75 seconds

## Summary of splitting at layer (split count: 11)
- Time for IS candidates: 32.14 seconds
IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 12, time: 32.14
Output dim: 7, lower bound: -1.5806746, upper bound: 1.6129009
IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 12, time: 32.14
Output dim: 7, lower bound: -1.5806746, upper bound: 1.6188198
IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 12, time: 32.14
Output dim: 7, lower bound: -1.5843205, upper bound: 1.6136063
IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 32.14
Output dim: 7, lower bound: -1.5843206, upper bound: 1.6193927
IS_A1_B2_A2_B2_A2_B2_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 12, time: 32.14
Output dim: 7, lower bound: -1.5883981, upper bound: 1.6188064
IS_A1_B2_A2_B2_A2_B2_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 12, time: 32.14
Output dim: 7, lower bound: -1.5902911, upper bound: 1.6188148
IS_A1_B2_A2_B2_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 11, time: 32.14
Output dim: 7, lower bound: -1.5865395, upper bound: 1.6193948
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=3.029996871948242
rel_dist={7: [-1.6196747972199932, 1.6196725960818332]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2418.81 seconds
