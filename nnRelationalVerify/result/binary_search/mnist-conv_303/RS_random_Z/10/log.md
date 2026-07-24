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
execution time: IAR + LP analysis = 13.94 + 34.13 = 48.07 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -2.3592026, upper bound: 2.3592013


# Binary Search by BASE starts (time budget: 3551.93 seconds, max iter: 100)

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
Binary search time: 197.80 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Relational Split (RS_random_Z) starts
Time budget: 3354.13 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 5856
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4682

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 565

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9887453, upper bound: 1.9962101
time: 4.98 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9887468, upper bound: 1.9887448
time: 5.03 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 10.02 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 10.02
Output dim: 7, lower bound: -1.9887453, upper bound: 1.9962101
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 10.02
Output dim: 7, lower bound: -1.9887468, upper bound: 1.9887448

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.5703955, 3.5478516
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.8496227, 2.8665214
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.8174796, 2.8033769
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.3991976, 3.3920240
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.3360763, 3.3310423
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.1132982, 3.1132982
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3615036, 2.3615036
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.5797234, 3.5682249

Time for backsubstitution: 14.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5856
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 4666

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5856

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9765336, upper bound: 1.9961678
time: 5.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9887031, upper bound: 1.9839899
time: 4.78 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.5478506, 3.5703959
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.8665218, 2.8496218
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.8033767, 2.8174794
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.3920240, 3.3991981
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.3310428, 3.3360763
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.1132982, 3.1132982
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3615036, 2.3615036
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.5682249, 3.5797238

Time for backsubstitution: 14.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 5856
type: RSZ, layer: 1, pos: 6139

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4666

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9961401, upper bound: 1.9887306
time: 4.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9961955, upper bound: 1.9886716
time: 5.30 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 24.32 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.32
Output dim: 7, lower bound: -1.9765336, upper bound: 1.9961678
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.32
Output dim: 7, lower bound: -1.9887031, upper bound: 1.9839899
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.32
Output dim: 7, lower bound: -1.9961401, upper bound: 1.9887306
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.32
Output dim: 7, lower bound: -1.9961955, upper bound: 1.9886716

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.5705247, 3.5475440
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.8497934, 2.8661242
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.8144298, 2.8046532
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.3978310, 3.3925939
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.3341308, 3.3318582
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.1132982, 3.1132982
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3615036, 2.3615036
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.5802603, 3.5669241

Time for backsubstitution: 14.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4666

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 944

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9764648, upper bound: 1.9900886
time: 4.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9704376, upper bound: 1.9961007
time: 4.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.5700879, 3.5478516
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.8492250, 2.8665214
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.8174796, 2.8003273
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.3991976, 3.3906569
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.3360763, 3.3290968
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.1132982, 3.1132982
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3615036, 2.3615036
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.5784235, 3.5682249

Time for backsubstitution: 14.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 4571

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 887

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9887003, upper bound: 1.9808961
time: 4.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9734293, upper bound: 1.9839878
time: 4.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.5444517, 3.5708065
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.8680239, 2.8375688
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.7974458, 2.8182023
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.3852224, 3.4000254
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.3243566, 3.3368948
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.1132982, 3.1132982
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3615036, 2.3615036
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.5658913, 3.5800085

Time for backsubstitution: 14.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 5856
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 6220

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4573

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9961252, upper bound: 1.9844625
time: 4.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9843493, upper bound: 1.9887163
time: 4.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.5478506, 3.5669956
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.8544683, 2.8496218
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.8033767, 2.8115485
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.3920240, 3.3923965
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.3310428, 3.3293903
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.1132982, 3.1132982
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3615036, 2.3615036
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.5682249, 3.5773897

Time for backsubstitution: 14.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 5856
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 6220

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9961911, upper bound: 1.9854979
time: 5.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9929606, upper bound: 1.9886675
time: 5.12 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 24.57 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.57
Output dim: 7, lower bound: -1.9764648, upper bound: 1.9900886
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.57
Output dim: 7, lower bound: -1.9704376, upper bound: 1.9961007
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.57
Output dim: 7, lower bound: -1.9887003, upper bound: 1.9808961
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.57
Output dim: 7, lower bound: -1.9734293, upper bound: 1.9839878
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.57
Output dim: 7, lower bound: -1.9961252, upper bound: 1.9844625
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.57
Output dim: 7, lower bound: -1.9843493, upper bound: 1.9887163
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.57
Output dim: 7, lower bound: -1.9961911, upper bound: 1.9854979
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.57
Output dim: 7, lower bound: -1.9929606, upper bound: 1.9886675

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.5723376, 3.5503130
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.8485699, 2.8625565
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.8141565, 2.8038568
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.3845510, 3.3738441
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.3345046, 3.3324304
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.1132982, 3.1132982
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3615036, 2.3615036
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.5711765, 3.5604935

Time for backsubstitution: 14.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4682

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5801

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9764596, upper bound: 1.9866347
time: 4.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9732135, upper bound: 1.9900832
time: 5.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.5732932, 3.5493569
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.8462257, 2.8649015
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.8136339, 2.8043771
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.3790808, 3.3793187
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.3347030, 3.3322327
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.1132982, 3.1132982
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3615036, 2.3615036
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.5738316, 3.5578418

Time for backsubstitution: 14.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 5801

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 160

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9704274, upper bound: 1.9935489
time: 4.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9704351, upper bound: 1.9861474
time: 5.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.5700750, 3.5477824
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.8461599, 2.8659568
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.8169498, 2.7974510
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.3987560, 3.3882179
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.3357120, 3.3290300
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.1132982, 3.1132982
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3615036, 2.3615036
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.5781651, 3.5668683

Time for backsubstitution: 14.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 5801

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4573

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9886862, upper bound: 1.9765733
time: 4.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9844317, upper bound: 1.9808819
time: 4.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.5700197, 3.5478373
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.8486595, 2.8634567
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.8146038, 2.7997980
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.3967590, 3.3902159
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.3360105, 3.3287320
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.1132982, 3.1132982
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3615036, 2.3615036
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.5770674, 3.5679655

Time for backsubstitution: 14.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 957

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4682

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9851555, upper bound: 1.9839862
time: 5.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9855774, upper bound: 1.9835630
time: 4.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.5414505, 3.5665751
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.8567157, 2.8199375
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.8082461, 2.8334126
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.3926039, 3.4130359
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.3406816, 3.3486245
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.1132982, 3.1132982
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3615036, 2.3615036
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.5716791, 3.5880756

Time for backsubstitution: 14.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 5856
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 5858

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 887

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9961229, upper bound: 1.9813454
time: 4.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9930542, upper bound: 1.9844596
time: 4.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.5402203, 3.5678062
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.8503919, 2.8262603
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.8126559, 2.8290019
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.3982325, 3.4074078
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.3360868, 3.3532200
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.1132982, 3.1132982
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3615036, 2.3615036
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.5739584, 3.5857964

Time for backsubstitution: 14.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 5856

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 887

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9918133, upper bound: 1.9855928
time: 6.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9888121, upper bound: 1.9887134
time: 4.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.5240774, 3.5334463
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.8296552, 2.8146157
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.7857304, 2.7990367
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.4018183, 3.4083605
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.3519111, 3.3441408
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.1132982, 3.1132982
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3615036, 2.3615036
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.5689430, 3.5784011

Time for backsubstitution: 14.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 5856
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 523

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6220

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9898310, upper bound: 1.9854873
time: 4.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9961808, upper bound: 1.9792600
time: 9.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.5143023, 3.5432215
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.8194623, 2.8248093
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.7908659, 2.7939017
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.4079866, 3.4021912
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.3457942, 3.3502591
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.1132982, 3.1132982
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3615036, 2.3615036
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.5692368, 3.5781069

Time for backsubstitution: 14.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 5856
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 6139

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4573

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9929472, upper bound: 1.9843435
time: 5.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9886239, upper bound: 1.9886523
time: 6.82 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 26.27 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.27
Output dim: 7, lower bound: -1.9764596, upper bound: 1.9866347
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.27
Output dim: 7, lower bound: -1.9732135, upper bound: 1.9900832
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.27
Output dim: 7, lower bound: -1.9704274, upper bound: 1.9935489
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.27
Output dim: 7, lower bound: -1.9704351, upper bound: 1.9861474
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.27
Output dim: 7, lower bound: -1.9886862, upper bound: 1.9765733
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.27
Output dim: 7, lower bound: -1.9844317, upper bound: 1.9808819
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.27
Output dim: 7, lower bound: -1.9851555, upper bound: 1.9839862
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.27
Output dim: 7, lower bound: -1.9855774, upper bound: 1.9835630
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.27
Output dim: 7, lower bound: -1.9961229, upper bound: 1.9813454
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.27
Output dim: 7, lower bound: -1.9930542, upper bound: 1.9844596
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.27
Output dim: 7, lower bound: -1.9918133, upper bound: 1.9855928
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.27
Output dim: 7, lower bound: -1.9888121, upper bound: 1.9887134
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.27
Output dim: 7, lower bound: -1.9898310, upper bound: 1.9854873
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.27
Output dim: 7, lower bound: -1.9961808, upper bound: 1.9792600
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.27
Output dim: 7, lower bound: -1.9929472, upper bound: 1.9843435
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.27
Output dim: 7, lower bound: -1.9886239, upper bound: 1.9886523

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.5725894, 3.5506954
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.8141870, 2.8381782
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.7956305, 2.7765231
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.3656693, 3.3472052
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.3248439, 3.3255799
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.1132982, 3.1132982
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3615036, 2.3615036
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.5639801, 3.5503507

Time for backsubstitution: 14.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 957

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6139

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9764583, upper bound: 1.9859536
time: 4.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9757428, upper bound: 1.9866334
time: 5.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.5727210, 3.5505652
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.8241911, 2.8281751
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.7868223, 2.7853308
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.3579130, 3.3549619
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.3276544, 3.3227687
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.1132982, 3.1132982
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3615036, 2.3560784
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.5610352, 3.5532951

Time for backsubstitution: 14.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4573

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5871

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9482730, upper bound: 1.9900781
time: 5.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9732084, upper bound: 1.9651364
time: 6.09 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.5732951, 3.5493588
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.8462243, 2.8649001
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.8136263, 2.8043671
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.3790798, 3.3793168
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.3347125, 3.3322434
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.1132982, 3.1132982
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3615036, 2.3615036
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.5738297, 3.5578384

Time for backsubstitution: 14.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4572

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4682

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9699963, upper bound: 1.9935481
time: 4.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9704257, upper bound: 1.9931291
time: 4.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.5732951, 3.5493598
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.8462243, 2.8648992
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.8136225, 2.8043709
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.3790789, 3.3793178
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.3347077, 3.3322425
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.1132982, 3.1132982
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3615036, 2.3615036
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.5738277, 3.5578365

Time for backsubstitution: 14.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9704350, upper bound: 1.9799389
time: 5.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9642282, upper bound: 1.9861472
time: 5.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.5670738, 3.5435505
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.8348532, 2.8483257
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.8277502, 2.8126621
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.4061394, 3.4012275
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.3520379, 3.3407600
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.1132982, 3.1132982
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3615036, 2.3615036
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.5839548, 3.5749364

Time for backsubstitution: 14.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 6220

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9886861, upper bound: 1.9703992
time: 4.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9824698, upper bound: 1.9765731
time: 5.04 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.5658417, 3.5447822
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.8285303, 2.8546486
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.8321609, 2.8082514
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.4117670, 3.3955998
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.3474422, 3.3453555
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.1132982, 3.1132982
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3615036, 2.3615036
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.5862341, 3.5726571

Time for backsubstitution: 14.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 523

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 160

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9839769, upper bound: 1.9805121
time: 4.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9839774, upper bound: 1.9731114
time: 4.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.5660686, 3.5422578
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.8709235, 2.8802447
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.8146787, 2.7997556
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.3762455, 3.3612347
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.3414779, 3.3359768
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.1132982, 3.1132982
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3615036, 2.3615036
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.5628624, 3.5579095

Time for backsubstitution: 14.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 4626

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4572

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9851417, upper bound: 1.9774973
time: 5.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9665416, upper bound: 1.9839725
time: 5.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.5644417, 3.5438857
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.8654485, 2.8857160
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.8145604, 2.7998729
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.3677778, 3.3697038
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.3432536, 3.3341990
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.1132982, 3.1132982
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3615036, 2.3615036
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.5670109, 3.5537605

Time for backsubstitution: 14.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 4572

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 944

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9855069, upper bound: 1.9774710
time: 5.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9794589, upper bound: 1.9834952
time: 6.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.5414367, 3.5665059
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.8536515, 2.8193734
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.8077173, 2.8305373
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.3921642, 3.4105968
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.3403182, 3.3485591
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.1132982, 3.1132982
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3615036, 2.3615036
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.5714207, 3.5867200

Time for backsubstitution: 14.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5856
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 4572

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5871

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9711742, upper bound: 1.9813402
time: 4.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9961177, upper bound: 1.9563812
time: 4.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.5413833, 3.5665607
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.8561511, 2.8168733
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.8053703, 2.8328838
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.3901653, 3.4125948
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.3406167, 3.3482609
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.1132982, 3.1132982
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3615036, 2.3615036
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.5703239, 3.5878172

Time for backsubstitution: 14.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 5856
type: RSZ, layer: 1, pos: 5858

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4626

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9930417, upper bound: 1.9799885
time: 4.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9885795, upper bound: 1.9844475
time: 5.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.5402064, 3.5677376
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.8473287, 2.8256962
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.8121281, 2.8261266
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.3977919, 3.4049687
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.3357234, 3.3531547
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.1132982, 3.1132982
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3615036, 2.3615036
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.5737000, 3.5844412

Time for backsubstitution: 14.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 5856
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 118

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4626

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9843423, upper bound: 1.9811192
time: 4.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9873425, upper bound: 1.9855805
time: 5.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.5401511, 3.5677919
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.8498282, 2.8231966
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.8097811, 2.8284736
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.3957939, 3.4069667
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.3360209, 3.3528564
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.1132982, 3.1132982
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3615036, 2.3615036
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.5726023, 3.5855379

Time for backsubstitution: 14.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 5856
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 4682

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5858

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9888093, upper bound: 1.9828264
time: 5.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9828967, upper bound: 1.9887122
time: 4.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.5161457, 3.5311480
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.8254056, 2.7997494
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.7838221, 2.7984600
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.4006710, 3.4080429
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.3480763, 3.3430212
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.1132982, 3.1132982
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3557994, 2.3615036
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.5602989, 3.5759182

Time for backsubstitution: 14.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5856
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 118

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 944

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9897636, upper bound: 1.9794562
time: 5.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9732566, upper bound: 1.9854166
time: 4.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.5217781, 3.5255146
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.8147817, 2.8103724
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.7851524, 2.7971277
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.4015036, 3.4072104
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.3507924, 3.3403046
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.1132982, 3.1132982
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3615036, 2.3615036
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.5664606, 3.5697570

Time for backsubstitution: 14.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 5856
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 4682

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4572

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9961665, upper bound: 1.9728065
time: 5.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9896924, upper bound: 1.9792485
time: 5.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.5113020, 3.5389900
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.8081632, 2.8071878
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.8016667, 2.8091135
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.4153719, 3.4152031
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.3621187, 3.3619885
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.1132982, 3.1132982
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3615036, 2.3615036
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.5750265, 3.5861759

Time for backsubstitution: 14.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 5856
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 5801

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5871

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9679871, upper bound: 1.9843383
time: 4.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9929421, upper bound: 1.9593997
time: 5.08 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.5100698, 3.5402241
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.8018403, 2.8135064
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.8060784, 2.8047032
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.4209948, 3.4095750
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.3575239, 3.3665853
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.1132982, 3.1132982
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3615036, 2.3615036
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.5773039, 3.5838966

Time for backsubstitution: 14.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 5856

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5858

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9886225, upper bound: 1.9827996
time: 4.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9827017, upper bound: 1.9886508
time: 5.06 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 23.94 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.94
Output dim: 7, lower bound: -1.9764583, upper bound: 1.9859536
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.94
Output dim: 7, lower bound: -1.9757428, upper bound: 1.9866334
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.94
Output dim: 7, lower bound: -1.9482730, upper bound: 1.9900781
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.94
Output dim: 7, lower bound: -1.9732084, upper bound: 1.9651364
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.94
Output dim: 7, lower bound: -1.9699963, upper bound: 1.9935481
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.94
Output dim: 7, lower bound: -1.9704257, upper bound: 1.9931291
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.94
Output dim: 7, lower bound: -1.9704350, upper bound: 1.9799389
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.94
Output dim: 7, lower bound: -1.9642282, upper bound: 1.9861472
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.94
Output dim: 7, lower bound: -1.9886861, upper bound: 1.9703992
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.94
Output dim: 7, lower bound: -1.9824698, upper bound: 1.9765731
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.94
Output dim: 7, lower bound: -1.9839769, upper bound: 1.9805121
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.94
Output dim: 7, lower bound: -1.9839774, upper bound: 1.9731114
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.94
Output dim: 7, lower bound: -1.9851417, upper bound: 1.9774973
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.94
Output dim: 7, lower bound: -1.9665416, upper bound: 1.9839725
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.94
Output dim: 7, lower bound: -1.9855069, upper bound: 1.9774710
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.94
Output dim: 7, lower bound: -1.9794589, upper bound: 1.9834952
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.94
Output dim: 7, lower bound: -1.9711742, upper bound: 1.9813402
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.94
Output dim: 7, lower bound: -1.9961177, upper bound: 1.9563812
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.94
Output dim: 7, lower bound: -1.9930417, upper bound: 1.9799885
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.94
Output dim: 7, lower bound: -1.9885795, upper bound: 1.9844475
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.94
Output dim: 7, lower bound: -1.9843423, upper bound: 1.9811192
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.94
Output dim: 7, lower bound: -1.9873425, upper bound: 1.9855805
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.94
Output dim: 7, lower bound: -1.9888093, upper bound: 1.9828264
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.94
Output dim: 7, lower bound: -1.9828967, upper bound: 1.9887122
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.94
Output dim: 7, lower bound: -1.9897636, upper bound: 1.9794562
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.94
Output dim: 7, lower bound: -1.9732566, upper bound: 1.9854166
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.94
Output dim: 7, lower bound: -1.9961665, upper bound: 1.9728065
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.94
Output dim: 7, lower bound: -1.9896924, upper bound: 1.9792485
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.94
Output dim: 7, lower bound: -1.9679871, upper bound: 1.9843383
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.94
Output dim: 7, lower bound: -1.9929421, upper bound: 1.9593997
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.94
Output dim: 7, lower bound: -1.9886225, upper bound: 1.9827996
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.94
Output dim: 7, lower bound: -1.9827017, upper bound: 1.9886508

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.5610952, 3.5344682
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.8053026, 2.8318887
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.7907686, 2.7730756
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.3618679, 3.3418350
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.2655134, 3.2835941
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.1132982, 3.1132982
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3615036, 2.3384829
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.5679350, 3.5583334

Time for backsubstitution: 13.99 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=3.113298177719116
rel_dist={7: [-1.996222388869347, 1.996221151991417]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5856
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 523

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5856

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7102930, upper bound: 1.7205079
time: 5.18 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7205075, upper bound: 1.7102924
time: 6.31 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 11.51 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 11.51
Output dim: 7, lower bound: -1.7102930, upper bound: 1.7205079
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 11.51
Output dim: 7, lower bound: -1.7205075, upper bound: 1.7102924

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.4650483, 3.4648004
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.7420006, 2.7416759
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6762557, 2.6787279
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.1529579, 3.1540647
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0986547, 3.1011891
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.1378942, 3.1394718
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0654659, 3.0675750
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3529110, 2.3525698
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3865709, 3.3855228

Time for backsubstitution: 14.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 6220

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 944

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7102241, upper bound: 1.7174117
time: 5.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7071911, upper bound: 1.7204402
time: 4.96 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.4648004, 3.4650478
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.7416763, 2.7420008
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6787276, 2.6762559
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.1540642, 3.1529579
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1011896, 3.0986543
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.1394715, 3.1378937
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0675745, 3.0654659
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3525701, 2.3529105
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3855228, 3.3865709

Time for backsubstitution: 14.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 4682

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7205074, upper bound: 1.7068779
time: 5.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7068788, upper bound: 1.7102902
time: 5.19 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 24.35 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.35
Output dim: 7, lower bound: -1.7102241, upper bound: 1.7174117
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.35
Output dim: 7, lower bound: -1.7071911, upper bound: 1.7204402
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.35
Output dim: 7, lower bound: -1.7205074, upper bound: 1.7068779
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.35
Output dim: 7, lower bound: -1.7068788, upper bound: 1.7102902

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.4668598, 3.4671578
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.7397699, 2.7381058
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6757565, 2.6779318
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.1373377, 3.1353168
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0935693, 3.0950866
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.1382680, 3.1399593
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0648165, 3.0665364
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3480828, 2.3467853
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3774862, 3.3779531

Time for backsubstitution: 14.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 4572

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5871

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6959886, upper bound: 1.7174082
time: 5.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7102206, upper bound: 1.7031973
time: 6.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.4674053, 3.4666114
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.7384310, 2.7394447
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6754599, 2.6782289
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.1342096, 3.1384449
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0925517, 3.0961046
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.1383805, 3.1398463
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0644274, 3.0669255
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3471258, 2.3477423
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3790016, 3.3764377

Time for backsubstitution: 14.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 5871

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7051901, upper bound: 1.7184364
time: 5.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7051901, upper bound: 1.7204374
time: 4.86 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.4647913, 3.4650369
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.7416592, 2.7419801
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6786985, 2.6762242
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.1540899, 3.1529794
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1011896, 3.0986543
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.1394596, 3.1378791
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0675483, 3.0654345
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3525372, 2.3528836
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3855219, 3.3865695

Time for backsubstitution: 14.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 887

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 565

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7023250, upper bound: 1.7068688
time: 5.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7204999, upper bound: 1.7023241
time: 5.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.4647894, 3.4650383
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.7416553, 2.7419837
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6786957, 2.6762271
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.1540861, 3.1529837
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1011887, 3.0986547
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.1394577, 3.1378813
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0675426, 3.0654383
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3525429, 2.3528783
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3855219, 3.3865700

Time for backsubstitution: 14.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 6220

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 67

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7170863, upper bound: 1.7085058
time: 5.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7050849, upper bound: 1.7102694
time: 7.00 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 26.25 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.25
Output dim: 7, lower bound: -1.6959886, upper bound: 1.7174082
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 26.25
Output dim: 7, lower bound: -1.7102206, upper bound: 1.7031973
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.25
Output dim: 7, lower bound: -1.7051901, upper bound: 1.7184364
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 26.25
Output dim: 7, lower bound: -1.7051901, upper bound: 1.7204374
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.25
Output dim: 7, lower bound: -1.7023250, upper bound: 1.7068688
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 26.25
Output dim: 7, lower bound: -1.7204999, upper bound: 1.7023241
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.25
Output dim: 7, lower bound: -1.7170863, upper bound: 1.7085058
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 26.25
Output dim: 7, lower bound: -1.7050849, upper bound: 1.7102694

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.4535789, 3.4512177
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.7459211, 2.7455561
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6706557, 2.6718106
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.1345062, 3.1336393
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0900154, 3.0945721
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.1368375, 3.1382427
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0599923, 3.0625157
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3284454, 2.3232181
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3682652, 3.3702669

Time for backsubstitution: 14.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 6139

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4571

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6959727, upper bound: 1.7133213
time: 5.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6919349, upper bound: 1.7173924
time: 5.14 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.4509182, 3.4538770
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.7472200, 2.7442570
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6696372, 2.6728306
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.1356602, 3.1324844
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0930557, 3.0915318
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.1365514, 3.1385288
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0607963, 3.0617118
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3245153, 2.3271475
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3698006, 3.3687315

Time for backsubstitution: 14.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4666

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5801

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7102160, upper bound: 1.7015682
time: 5.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7085399, upper bound: 1.7031928
time: 5.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.4394417, 3.4330621
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.7092490, 2.7044384
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6578131, 2.6635160
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.1440029, 3.1517639
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0782814, 3.0852981
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.1566267, 3.1545966
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0696445, 3.0733814
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3442373, 2.3442776
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3797207, 3.3773246

Time for backsubstitution: 14.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 4572

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 67

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7071664, upper bound: 1.7166543
time: 5.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7034072, upper bound: 1.7184131
time: 5.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.4338560, 3.4386477
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.7034240, 2.7102635
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6607475, 2.6605816
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.1475296, 3.1482391
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0817451, 3.0818338
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.1531305, 3.1580927
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0708833, 3.0721431
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3436613, 2.3448532
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3798885, 3.3771567

Time for backsubstitution: 14.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 957

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5858

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7046036, upper bound: 1.7168218
time: 5.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7046015, upper bound: 1.7204369
time: 4.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.4003849, 3.3877482
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6896439, 2.6996229
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6779575, 2.6674232
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.1335640, 3.1283531
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9590540, 3.9629693
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.1250715, 3.1206145
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0521111, 3.0525694
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.2669930, 2.2490151
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3848886, 3.3793664

Time for backsubstitution: 14.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 67

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5858

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7159276, upper bound: 1.7038932
time: 5.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7131017, upper bound: 1.7068684
time: 5.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.3875027, 3.4006324
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6993008, 2.6899655
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6698971, 2.6754818
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.1294632, 3.1324530
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9652948, 3.9567280
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.1221943, 3.1234910
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0546832, 3.0499973
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.2486691, 2.2673390
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3783178, 3.3859382

Time for backsubstitution: 14.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 4573

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 523

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7204984, upper bound: 1.7022891
time: 5.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7170678, upper bound: 1.7023217
time: 5.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.4647579, 3.4650116
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.7416553, 2.7419839
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6786990, 2.6762302
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.1540823, 3.1529803
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1011567, 3.0986166
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.1394310, 3.1378596
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0675292, 3.0654263
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3525367, 2.3528714
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3855152, 3.3865643

Time for backsubstitution: 14.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 4571

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5801

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7170812, upper bound: 1.7068114
time: 6.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7152118, upper bound: 1.7085029
time: 5.16 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.4647617, 3.4650068
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.7416553, 2.7419839
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6786990, 2.6762297
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.1540823, 3.1529799
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1011510, 3.0986223
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.1394348, 3.1378555
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0675311, 3.0654244
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3525357, 2.3528726
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3855152, 3.3865633

Time for backsubstitution: 14.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 5801

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 160

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7020355, upper bound: 1.7095074
time: 5.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7145568, upper bound: 1.7072113
time: 5.08 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 24.28 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.28
Output dim: 7, lower bound: -1.6959727, upper bound: 1.7133213
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.28
Output dim: 7, lower bound: -1.6919349, upper bound: 1.7173924
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.28
Output dim: 7, lower bound: -1.7102160, upper bound: 1.7015682
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.28
Output dim: 7, lower bound: -1.7085399, upper bound: 1.7031928
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.28
Output dim: 7, lower bound: -1.7071664, upper bound: 1.7166543
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.28
Output dim: 7, lower bound: -1.7034072, upper bound: 1.7184131
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.28
Output dim: 7, lower bound: -1.7046036, upper bound: 1.7168218
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.28
Output dim: 7, lower bound: -1.7046015, upper bound: 1.7204369
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.28
Output dim: 7, lower bound: -1.7159276, upper bound: 1.7038932
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.28
Output dim: 7, lower bound: -1.7131017, upper bound: 1.7068684
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.28
Output dim: 7, lower bound: -1.7204984, upper bound: 1.7022891
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.28
Output dim: 7, lower bound: -1.7170678, upper bound: 1.7023217
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.28
Output dim: 7, lower bound: -1.7170812, upper bound: 1.7068114
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.28
Output dim: 7, lower bound: -1.7152118, upper bound: 1.7085029
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.28
Output dim: 7, lower bound: -1.7020355, upper bound: 1.7095074
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.28
Output dim: 7, lower bound: -1.7145568, upper bound: 1.7072113

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.4536915, 3.4512172
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.7168932, 2.7213666
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6684890, 2.6692123
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.0969372, 3.0885501
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9589691, 3.9697371
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0907059, 3.0954475
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.0690598, 3.0817676
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0400896, 3.0382953
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3111446, 2.3024514
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3690443, 3.3702650

Time for backsubstitution: 14.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 5858

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4626

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6959645, upper bound: 1.7106152
time: 4.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6932168, upper bound: 1.7133150
time: 5.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.4535780, 3.4513311
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.7217321, 2.7165277
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6680579, 2.6696444
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.0894165, 3.0960717
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9653549, 3.9633517
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0908909, 3.0952640
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.0803628, 3.0704656
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0357714, 3.0426149
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3076785, 2.3059175
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3682623, 3.3710465

Time for backsubstitution: 14.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 6140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4626

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6919266, upper bound: 1.7146810
time: 5.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6891823, upper bound: 1.7173858
time: 5.26 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.4511719, 3.4542041
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.7128396, 2.7155924
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6473365, 2.6454973
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.1134539, 3.1058455
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9551067, 3.9501987
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0910730, 3.0891542
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.1268916, 3.1304741
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0580153, 3.0549893
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3036919, 2.3102343
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3613415, 3.3585892

Time for backsubstitution: 14.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 565

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4626

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7102078, upper bound: 1.6988871
time: 5.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7074693, upper bound: 1.7015598
time: 5.09 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.4512463, 3.4541292
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.7185559, 2.7098763
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6423030, 2.6505303
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.1090212, 3.1102777
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9494247, 3.9558802
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0906773, 3.0895505
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.1284976, 3.1288676
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0540729, 3.0589318
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3076029, 2.3063233
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3596582, 3.3602724

Time for backsubstitution: 14.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 6139

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 957

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6888768, upper bound: 1.6978514
time: 5.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7066070, upper bound: 1.6978168
time: 5.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.4394112, 3.4330354
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.7092495, 2.7044380
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6578155, 2.6635199
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.1440001, 3.1517611
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0782485, 3.0852604
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.1566010, 3.1545744
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0696311, 3.0733705
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3442302, 2.3442698
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3797131, 3.3773189

Time for backsubstitution: 14.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 4666

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4571

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6993499, upper bound: 1.7125960
time: 5.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6993499, upper bound: 1.7166385
time: 7.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.4394150, 3.4330311
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.7092495, 2.7044380
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6578164, 2.6635194
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.1440010, 3.1517611
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0782428, 3.0852656
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.1566057, 3.1545701
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0696330, 3.0733685
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3442287, 2.3442709
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3797140, 3.3773184

Time for backsubstitution: 14.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 4666

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 565

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6988911, upper bound: 1.7184056
time: 5.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7053823, upper bound: 1.7138312
time: 5.25 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.4407482, 3.4470110
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.7022629, 2.7039521
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6666303, 2.6644001
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.0956020, 3.0859351
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0733824, 3.0717950
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.1460905, 3.1499171
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0705481, 3.0715489
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3198614, 2.3163083
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3589163, 3.3622856

Time for backsubstitution: 14.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6139

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7046002, upper bound: 1.7161107
time: 5.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7038924, upper bound: 1.7168195
time: 5.06 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.4422197, 3.4455400
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6971130, 2.7091007
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6645656, 2.6664672
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.0852251, 3.0963106
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0717058, 3.0734706
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.1449556, 3.1512768
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0708036, 3.0718083
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3151164, 2.3210599
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3650179, 3.3561850

Time for backsubstitution: 14.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 67

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6139

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7045981, upper bound: 1.7197357
time: 5.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7038904, upper bound: 1.7204346
time: 5.19 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.4072762, 3.3961101
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6881104, 2.6929390
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6820583, 2.6694517
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.0816431, 3.0660520
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9589796, 3.9629073
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1098585, 3.1136022
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.1182556, 3.1124382
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0517740, 3.0524898
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.2432218, 2.2204850
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3588562, 3.3594351

Time for backsubstitution: 14.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4666

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7153178, upper bound: 1.7038828
time: 5.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7159189, upper bound: 1.7031337
time: 5.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.4087439, 3.3946390
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6829605, 2.6980844
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6799870, 2.6715174
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.0712624, 3.0764289
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9589911, 3.9628949
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1081829, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.1168957, 3.1137977
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0520296, 3.0522332
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.2384634, 2.2252297
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3649502, 3.3533335

Time for backsubstitution: 14.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 6140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4573

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7130912, upper bound: 1.7043461
time: 5.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7105719, upper bound: 1.7068583
time: 5.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.3905916, 3.4032388
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6576090, 2.6399288
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6671019, 2.6721084
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.1071033, 3.1138172
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9661970, 3.9577971
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1029367, 3.0865741
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.1135988, 3.1163268
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0384426, 3.0303354
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.2287431, 2.2538762
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3698483, 3.3793020

Time for backsubstitution: 14.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 4682

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 160

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7151050, upper bound: 1.7014442
time: 4.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7197235, upper bound: 1.7014442
time: 5.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.3901091, 3.4037218
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6492643, 2.6482747
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6665249, 2.6726825
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.1108189, 3.1100917
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9663620, 3.9576297
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0970650, 3.0924459
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.1150255, 3.1148946
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0350208, 3.0337563
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.2352061, 2.2474132
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3716812, 3.3774676

Time for backsubstitution: 14.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 944

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7169985, upper bound: 1.6992237
time: 5.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7138967, upper bound: 1.7022547
time: 5.45 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.4650116, 3.4653392
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.7072749, 2.7133191
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6563997, 2.6488972
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.1318760, 3.1263413
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9679775, 3.9599724
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0991740, 3.0962377
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.1297703, 3.1298048
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0647483, 3.0587025
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3317137, 2.3359592
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3770552, 3.3764215

Time for backsubstitution: 14.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4626

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6220

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7137722, upper bound: 1.7068062
time: 5.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7170765, upper bound: 1.7034354
time: 5.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.4650860, 3.4652648
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.7129912, 2.7076030
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6513662, 2.6539311
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.1274433, 3.1307745
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9622965, 3.9656544
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0987782, 3.0966339
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.1313763, 3.1281984
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0608058, 3.0626459
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3356247, 2.3320482
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3753729, 3.3781042

Time for backsubstitution: 14.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 4666

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 160

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7121175, upper bound: 1.7077254
time: 4.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7144174, upper bound: 1.7054237
time: 5.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.4647641, 3.4650087
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.7416573, 2.7419848
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6786933, 2.6762216
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.1540813, 3.1529784
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1011477, 3.0986199
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.1394463, 3.1378660
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0675325, 3.0654263
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3525391, 2.3528750
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3855133, 3.3865614

Time for backsubstitution: 14.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 957

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 944

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7121189, upper bound: 1.7068636
time: 5.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6994241, upper bound: 1.7093774
time: 5.22 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.4647641, 3.4650092
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.7416573, 2.7419841
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6786914, 2.6762235
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.1540813, 3.1529784
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1011496, 3.0986190
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.1394463, 3.1378655
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0675335, 3.0654259
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3525381, 2.3528762
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3855133, 3.3865604

Time for backsubstitution: 14.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4572

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7145538, upper bound: 1.7052008
time: 4.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7125518, upper bound: 1.7072082
time: 5.06 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 24.01 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.01
Output dim: 7, lower bound: -1.6959645, upper bound: 1.7106152
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.01
Output dim: 7, lower bound: -1.6932168, upper bound: 1.7133150
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.01
Output dim: 7, lower bound: -1.6919266, upper bound: 1.7146810
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.01
Output dim: 7, lower bound: -1.6891823, upper bound: 1.7173858
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.01
Output dim: 7, lower bound: -1.7102078, upper bound: 1.6988871
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.01
Output dim: 7, lower bound: -1.7074693, upper bound: 1.7015598
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.01
Output dim: 7, lower bound: -1.6888768, upper bound: 1.6978514
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.01
Output dim: 7, lower bound: -1.7066070, upper bound: 1.6978168
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.01
Output dim: 7, lower bound: -1.6993499, upper bound: 1.7125960
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.01
Output dim: 7, lower bound: -1.6993499, upper bound: 1.7166385
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.01
Output dim: 7, lower bound: -1.6988911, upper bound: 1.7184056
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.01
Output dim: 7, lower bound: -1.7053823, upper bound: 1.7138312
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.01
Output dim: 7, lower bound: -1.7046002, upper bound: 1.7161107
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.01
Output dim: 7, lower bound: -1.7038924, upper bound: 1.7168195
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.01
Output dim: 7, lower bound: -1.7045981, upper bound: 1.7197357
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.01
Output dim: 7, lower bound: -1.7038904, upper bound: 1.7204346
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.01
Output dim: 7, lower bound: -1.7153178, upper bound: 1.7038828
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.01
Output dim: 7, lower bound: -1.7159189, upper bound: 1.7031337
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.01
Output dim: 7, lower bound: -1.7130912, upper bound: 1.7043461
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.01
Output dim: 7, lower bound: -1.7105719, upper bound: 1.7068583
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.01
Output dim: 7, lower bound: -1.7151050, upper bound: 1.7014442
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.01
Output dim: 7, lower bound: -1.7197235, upper bound: 1.7014442
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.01
Output dim: 7, lower bound: -1.7169985, upper bound: 1.6992237
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.01
Output dim: 7, lower bound: -1.7138967, upper bound: 1.7022547
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.01
Output dim: 7, lower bound: -1.7137722, upper bound: 1.7068062
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.01
Output dim: 7, lower bound: -1.7170765, upper bound: 1.7034354
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.01
Output dim: 7, lower bound: -1.7121175, upper bound: 1.7077254
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.01
Output dim: 7, lower bound: -1.7144174, upper bound: 1.7054237
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.01
Output dim: 7, lower bound: -1.7121189, upper bound: 1.7068636
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.01
Output dim: 7, lower bound: -1.6994241, upper bound: 1.7093774
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.01
Output dim: 7, lower bound: -1.7145538, upper bound: 1.7052008
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.01
Output dim: 7, lower bound: -1.7125518, upper bound: 1.7072082
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=3.0680713653564453
rel_dist={7: [-1.720529757154063, 1.7205290576462247]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 5856
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 4573

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4571

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6196597, upper bound: 1.6165885
time: 5.33 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6165904, upper bound: 1.6196590
time: 4.80 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 10.15 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 10.15
Output dim: 7, lower bound: -1.6196597, upper bound: 1.6165885
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 10.15
Output dim: 7, lower bound: -1.6165904, upper bound: 1.6196590

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.4118490, 3.4117637
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6598492, 2.6634786
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6327395, 2.6324155
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.0274363, 3.0217953
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9129362, 3.9177246
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0454135, 3.0455513
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.0025711, 3.0110488
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0090170, 3.0057774
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.2901607, 2.2875612
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3245392, 3.3239527

Time for backsubstitution: 14.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 5856
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 4626

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 887

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6196578, upper bound: 1.6152333
time: 5.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.6152357, upper bound: 1.6165885
time: 5.64 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.4117651, 3.4118495
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6634789, 2.6598494
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6324153, 2.6327398
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.0217953, 3.0274363
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9177256, 3.9129357
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0455518, 3.0454135
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.0110493, 3.0025721
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0057774, 3.0090170
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.2875609, 2.2901607
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3239527, 3.3245387

Time for backsubstitution: 14.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 5856
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6220

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6143596, upper bound: 1.6196545
time: 4.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.6143596, upper bound: 1.6174241
time: 5.09 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 23.77 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 23.77
Output dim: 7, lower bound: -1.6196578, upper bound: 1.6152333
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 23.77
Output dim: 7, lower bound: -1.6152357, upper bound: 1.6165885
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 23.77
Output dim: 7, lower bound: -1.6143596, upper bound: 1.6196545
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 23.77
Output dim: 7, lower bound: -1.6143596, upper bound: 1.6174241

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.4118042, 3.4116945
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6567850, 2.6614854
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6308703, 2.6295402
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.0258541, 3.0193567
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9099445, 3.9131269
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0448837, 3.0447354
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.0022078, 3.0108123
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0077410, 3.0038104
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.2884917, 2.2864783
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3236523, 3.3225961

Time for backsubstitution: 14.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5856
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 5871

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5856

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.6075801, upper bound: 1.6152167
time: 5.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.6075801, upper bound: 1.6075776
time: 5.27 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.4038324, 3.4063315
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6531663, 2.6449831
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6305065, 2.6314011
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.0206470, 3.0266447
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9131432, 3.9063268
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0388174, 3.0357356
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.0072136, 2.9999001
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0020828, 3.0064559
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.2683320, 2.2768154
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3153086, 3.3185349

Time for backsubstitution: 14.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 5856
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 5858

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 67

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.6129738, upper bound: 1.6182948
time: 5.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6129738, upper bound: 1.6196356
time: 4.58 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 23.65 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 23.65
Output dim: 7, lower bound: -1.6075801, upper bound: 1.6152167
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 23.65
Output dim: 7, lower bound: -1.6075801, upper bound: 1.6075776
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 23.65
Output dim: 7, lower bound: -1.6129738, upper bound: 1.6182948
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.65
Output dim: 7, lower bound: -1.6129738, upper bound: 1.6196356

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.4038048, 3.4063005
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6531649, 2.6449835
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6305094, 2.6314032
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.0206442, 3.0266423
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9131441, 3.9063263
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0387816, 3.0357037
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.0071907, 2.9998751
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0020695, 3.0064421
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.2683253, 2.2768095
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3153019, 3.3185277

Time for backsubstitution: 14.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 5856

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4626

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.6110411, upper bound: 1.6177285
time: 10.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6110411, upper bound: 1.6196292
time: 5.07 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 29.26 seconds
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 29.26
Output dim: 7, lower bound: -1.6110411, upper bound: 1.6177285
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.26
Output dim: 7, lower bound: -1.6110411, upper bound: 1.6196292

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.4080439, 3.4114351
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6527929, 2.6481841
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6253800, 2.6269116
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.0073924, 3.0114999
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9106998, 3.9035392
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0393205, 3.0363560
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.0039144, 2.9961348
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0006452, 3.0051947
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.2600808, 2.2659512
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3040543, 3.3056774

Time for backsubstitution: 13.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 5856
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 887

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.6095441, upper bound: 1.6181348
time: 5.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6095441, upper bound: 1.6196272
time: 5.12 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 24.20 seconds
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.20
Output dim: 7, lower bound: -1.6095441, upper bound: 1.6181348
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.20
Output dim: 7, lower bound: -1.6095441, upper bound: 1.6196272

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.3744960, 3.3820758
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6177769, 2.6175365
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6099329, 2.6092641
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.0198307, 3.0212951
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9148378, 3.9084120
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0276470, 3.0220847
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.0186644, 3.0135074
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0067892, 3.0104103
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.2566168, 2.2629187
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3048983, 3.3063960

Time for backsubstitution: 14.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 5856
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 957

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 944

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.6078449, upper bound: 1.6178085
time: 5.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6078449, upper bound: 1.6195666
time: 5.16 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 24.50 seconds
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 24.50
Output dim: 7, lower bound: -1.6078449, upper bound: 1.6178085
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 24.50
Output dim: 7, lower bound: -1.6078449, upper bound: 1.6195666

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.3767180, 3.3838882
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6142054, 2.6149704
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6091380, 2.6086922
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.0010815, 3.0048923
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9135036, 3.9068890
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0215464, 3.0167460
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.0191250, 3.0138824
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0057507, 3.0096631
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.2508316, 2.2578514
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.2969503, 3.2973118

Time for backsubstitution: 14.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 5856
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 118

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5858

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.6074105, upper bound: 1.6172359
time: 5.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6074112, upper bound: 1.6195678
time: 5.05 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 24.55 seconds
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 24.55
Output dim: 7, lower bound: -1.6074105, upper bound: 1.6172359
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 24.55
Output dim: 7, lower bound: -1.6074112, upper bound: 1.6195678

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.3847141, 3.3907809
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6078939, 2.6125200
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6129565, 2.6140604
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -2.9387779, 2.9503698
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9134398, 3.9068146
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0115080, 3.0079651
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.0109482, 3.0067258
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0056057, 3.0093265
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.2222867, 2.2328691
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.2805548, 3.2763400

Time for backsubstitution: 14.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 5856
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 4573

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 565

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6037525, upper bound: 1.6195616
time: 5.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.6074037, upper bound: 1.6158367
time: 5.05 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 24.48 seconds
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 24.48
Output dim: 7, lower bound: -1.6037525, upper bound: 1.6195616
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 24.48
Output dim: 7, lower bound: -1.6074037, upper bound: 1.6158367

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.3170834, 3.3134913
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.5558796, 2.5677459
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6102037, 2.6052637
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -2.9172306, 2.9257488
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.8759966, 3.8740525
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0285401, 3.0309629
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -2.9958429, 2.9894629
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -2.9901686, 2.9958186
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.1321712, 2.1290104
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.2782707, 3.2691326

Time for backsubstitution: 14.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 5856
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5871

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.5929139, upper bound: 1.6195581
time: 5.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.5929139, upper bound: 1.6087221
time: 5.24 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 24.36 seconds
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 24.36
Output dim: 7, lower bound: -1.5929139, upper bound: 1.6195581
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 24.36
Output dim: 7, lower bound: -1.5929139, upper bound: 1.6087221

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.3031373, 3.2975502
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.5620313, 2.5748727
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6048479, 2.5991440
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -2.9144568, 2.9238439
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.8651743, 3.8645840
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0249777, 3.0296817
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -2.9943409, 2.9877467
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -2.9853454, 2.9915986
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.1115494, 2.1054418
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.2690487, 3.2610621

Time for backsubstitution: 14.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 5856
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 4682

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4572

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.5902946, upper bound: 1.6169571
time: 5.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.5902946, upper bound: 1.6195526
time: 5.08 seconds

## Summary of splitting (split count: 9)
- Time for RS candidates: 24.36 seconds
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 24.36
Output dim: 7, lower bound: -1.5902946, upper bound: 1.6169571
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 10, time: 24.36
Output dim: 7, lower bound: -1.5902946, upper bound: 1.6195526

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.2948556, 3.2903028
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.5657778, 2.5792186
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6067200, 2.6000824
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -2.9211073, 2.9295712
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.8653841, 3.8648162
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0293093, 3.0334110
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -2.9920964, 2.9851828
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -2.9871163, 2.9931235
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.1111116, 2.1050584
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.2711735, 3.2635322

Time for backsubstitution: 14.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5856
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4573

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5856

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.5826411, upper bound: 1.6195338
time: 4.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.5826411, upper bound: 1.6118962
time: 5.70 seconds

## Summary of splitting (split count: 10)
- Time for RS candidates: 24.74 seconds
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 11, time: 24.74
Output dim: 7, lower bound: -1.5826411, upper bound: 1.6195338
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 11, time: 24.74
Output dim: 7, lower bound: -1.5826411, upper bound: 1.6118962

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.2947359, 3.2899952
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.5656242, 2.5788209
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6036692, 2.5988863
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -2.9197407, 2.9290357
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.8625154, 3.8636909
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0261784, 3.0321822
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -2.9901509, 2.9844205
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -2.9845109, 2.9921007
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.1109471, 2.1046381
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.2706614, 3.2622328

Time for backsubstitution: 14.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 4573

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6139

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.5826371, upper bound: 1.6190254
time: 5.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.5821174, upper bound: 1.6195316
time: 5.66 seconds

## Summary of splitting (split count: 11)
- Time for RS candidates: 24.93 seconds
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 12, time: 24.93
Output dim: 7, lower bound: -1.5826371, upper bound: 1.6190254
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 12, time: 24.93
Output dim: 7, lower bound: -1.5821174, upper bound: 1.6195316

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.2805371, 3.2737679
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.5567379, 2.5710461
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.5988102, 2.5946331
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -2.9150405, 2.9236646
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.8297958, 3.8350625
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0437698, 3.0527420
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -2.9307976, 2.9325001
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -2.9880290, 2.9939265
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.0895612, 2.0801980
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.2746210, 3.2679181

Time for backsubstitution: 14.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 887

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 957

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.5776680, upper bound: 1.6172637
time: 5.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.5782058, upper bound: 1.6037542
time: 5.17 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.2785087, 3.2757955
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.5578461, 2.5699351
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.5994167, 2.5940266
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -2.9143691, 2.9243283
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.8338814, 3.8309712
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0467415, 3.0497737
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -2.9382143, 2.9250677
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -2.9863381, 2.9956126
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.0865066, 2.0832479
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.2763472, 3.2661915

Time for backsubstitution: 14.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 887

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4666

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.5818278, upper bound: 1.6195209
time: 5.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.5821074, upper bound: 1.6192206
time: 5.64 seconds

## Summary of splitting (split count: 12)
- Time for RS candidates: 25.00 seconds
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 13, time: 25.00
Output dim: 7, lower bound: -1.5776680, upper bound: 1.6172637
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 13, time: 25.00
Output dim: 7, lower bound: -1.5782058, upper bound: 1.6037542
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 13, time: 25.00
Output dim: 7, lower bound: -1.5818278, upper bound: 1.6195209
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 13, time: 25.00
Output dim: 7, lower bound: -1.5821074, upper bound: 1.6192206

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.2751079, 3.2740283
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.5515885, 2.5578711
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.5934849, 2.5909467
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -2.9075656, 2.9207945
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.8338604, 3.8309321
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0448723, 3.0488014
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -2.9315281, 2.9215946
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -2.9847555, 2.9925556
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.0835893, 2.0776284
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.2740126, 3.2649779

Time for backsubstitution: 14.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 523

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 887

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.5818243, upper bound: 1.6181575
time: 5.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.5804139, upper bound: 1.6195184
time: 5.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.2767415, 3.2723947
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.5457826, 2.5636797
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.5963364, 2.5880957
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -2.9108348, 2.9175239
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.8338413, 3.8309512
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0457668, 3.0479040
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -2.9347439, 2.9183812
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -2.9832811, 2.9940295
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.0808876, 2.0803320
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.2751341, 3.2638564

Time for backsubstitution: 14.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 118

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5801

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.5806965, upper bound: 1.6178527
time: 5.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.5806964, upper bound: 1.6192166
time: 5.30 seconds

## Summary of splitting (split count: 13)
- Time for RS candidates: 24.83 seconds
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 14, time: 24.83
Output dim: 7, lower bound: -1.5818243, upper bound: 1.6181575
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 14, time: 24.83
Output dim: 7, lower bound: -1.5804139, upper bound: 1.6195184
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 14, time: 24.83
Output dim: 7, lower bound: -1.5806965, upper bound: 1.6178527
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 14, time: 24.83
Output dim: 7, lower bound: -1.5806964, upper bound: 1.6192166

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.2750387, 3.2739830
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.5495944, 2.5548055
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.5906096, 2.5890789
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -2.9051285, 2.9192128
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.8292646, 3.8279414
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0440555, 3.0482717
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -2.9312916, 2.9212303
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -2.9827886, 2.9912801
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.0825078, 2.0759611
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.2726564, 3.2640929

Time for backsubstitution: 14.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 4682

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.5804139, upper bound: 1.6169827
time: 5.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.5778488, upper bound: 1.6195190
time: 5.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.2770500, 3.2726474
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.5156813, 2.5292916
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.5690041, 2.5645390
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -2.8841991, 2.8942118
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.7996445, 3.8010144
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0433884, 3.0458217
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -2.9262872, 2.9087195
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -2.9765587, 2.9902644
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.0629969, 2.0595093
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.2649937, 3.2549772

Time for backsubstitution: 14.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 4682

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 523

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.5806921, upper bound: 1.6158562
time: 5.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.5773612, upper bound: 1.6192141
time: 7.52 seconds

## Summary of splitting (split count: 14)
- Time for RS candidates: 26.87 seconds
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 15, time: 26.87
Output dim: 7, lower bound: -1.5804139, upper bound: 1.6169827
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 15, time: 26.87
Output dim: 7, lower bound: -1.5778488, upper bound: 1.6195190
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 15, time: 26.87
Output dim: 7, lower bound: -1.5806921, upper bound: 1.6158562
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 15, time: 26.87
Output dim: 7, lower bound: -1.5773612, upper bound: 1.6192141

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.2750297, 3.2739744
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.5495744, 2.5547869
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.5905790, 2.5890505
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -2.9051485, 2.9192367
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.8292704, 3.8279476
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0440578, 3.0482745
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -2.9312778, 2.9212179
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -2.9827571, 2.9912519
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.0824802, 2.0759296
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.2726536, 3.2640905

Time for backsubstitution: 14.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 957

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 523

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.5770982, upper bound: 1.6161562
time: 5.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.5770646, upper bound: 1.6195158
time: 5.22 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.2796574, 3.2756162
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.4656105, 2.4854779
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.5656018, 2.5615683
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -2.8646240, 2.8718433
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.8006706, 3.8019176
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0142779, 3.0211129
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -2.9187632, 2.9001241
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -2.9568663, 2.9731374
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.0478597, 2.0395246
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.2578993, 3.2465076

Time for backsubstitution: 14.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 4682

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.5773603, upper bound: 1.6158826
time: 5.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.5773566, upper bound: 1.6192136
time: 5.34 seconds

## Summary of splitting (split count: 15)
- Time for RS candidates: 24.69 seconds
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 16, time: 24.69
Output dim: 7, lower bound: -1.5770982, upper bound: 1.6161562
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 16, time: 24.69
Output dim: 7, lower bound: -1.5770646, upper bound: 1.6195158
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 16, time: 24.69
Output dim: 7, lower bound: -1.5773603, upper bound: 1.6158826
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 16, time: 24.69
Output dim: 7, lower bound: -1.5773566, upper bound: 1.6192136

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.2776365, 3.2769432
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.4995031, 2.5109732
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.5872054, 2.5861082
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -2.8855748, 2.8968697
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.8302994, 3.8288522
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0149431, 3.0235620
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -2.9237542, 2.9126210
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -2.9630957, 2.9741573
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.0673871, 2.0559900
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.2655602, 3.2556214

Time for backsubstitution: 14.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 4682

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 957

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.5725838, upper bound: 1.6177527
time: 5.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.5726016, upper bound: 1.6042440
time: 5.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.2796469, 3.2756081
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.4655886, 2.4854591
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.5656033, 2.5615726
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -2.8646460, 2.8718686
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.8006783, 3.8019233
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0142765, 3.0211129
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -2.9187498, 2.9001095
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -2.9568644, 2.9731407
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.0478830, 2.0395448
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.2578993, 3.2465067

Time for backsubstitution: 13.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 887

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4682

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.5760604, upper bound: 1.6192110
time: 5.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.5773532, upper bound: 1.6179738
time: 5.23 seconds

## Summary of splitting (split count: 16)
- Time for RS candidates: 24.51 seconds
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 17, time: 24.51
Output dim: 7, lower bound: -1.5725838, upper bound: 1.6177527
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 17, time: 24.51
Output dim: 7, lower bound: -1.5726016, upper bound: 1.6042440
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 17, time: 24.51
Output dim: 7, lower bound: -1.5760604, upper bound: 1.6192110
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 17, time: 24.51
Output dim: 7, lower bound: -1.5773532, upper bound: 1.6179738

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.2747660, 3.2700286
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.4847202, 2.5022435
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.5656109, 2.5615301
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -2.8392816, 2.8428764
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.7924442, 3.7947192
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0066395, 3.0123839
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -2.9242191, 2.9063394
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -2.9582047, 2.9743166
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.0407610, 2.0314033
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.2436895, 3.2340755

Time for backsubstitution: 14.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 887

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.5760550, upper bound: 1.6178425
time: 5.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.5760545, upper bound: 1.6192085
time: 5.48 seconds

## Summary of splitting (split count: 17)
- Time for RS candidates: 25.00 seconds
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 18, time: 25.00
Output dim: 7, lower bound: -1.5760550, upper bound: 1.6178425
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 18, time: 25.00
Output dim: 7, lower bound: -1.5760545, upper bound: 1.6192085

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.2746968, 3.2699833
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.4827309, 2.4991782
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.5656042, 2.5625319
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -2.8368435, 2.8412938
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.7878475, 3.7917366
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0058246, 3.0118551
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -2.9239817, 2.9059746
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -2.9581990, 2.9750023
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.0396795, 2.0297360
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.2423344, 3.2331953

Time for backsubstitution: 14.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 4573

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 160

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.5746625, upper bound: 1.6180289
time: 5.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.5746625, upper bound: 1.6143237
time: 5.70 seconds

## Summary of splitting (split count: 18)
- Time for RS candidates: 25.49 seconds
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 19, time: 25.49
Output dim: 7, lower bound: -1.5746625, upper bound: 1.6180289
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 19, time: 25.49
Output dim: 7, lower bound: -1.5746625, upper bound: 1.6143237
Binary search (step 2): status=Status.VERIFIED, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=3.029996871948242
rel_dist={7: [-1.6196747972199932, 1.6196725960818332]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01171875
execution time: 2297.28 seconds
