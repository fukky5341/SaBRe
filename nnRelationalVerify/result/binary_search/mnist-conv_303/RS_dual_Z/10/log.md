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
execution time: IAR + LP analysis = 13.84 + 34.18 = 48.02 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -2.3592026, upper bound: 2.3592013


# Binary Search by BASE starts (time budget: 3551.98 seconds, max iter: 100)

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
Binary search time: 197.55 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Relational Split (RS_dual_Z) starts
Time budget: 3354.43 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 5856
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 565

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9887453, upper bound: 1.9962101
time: 4.84 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9962097, upper bound: 1.9887454
time: 5.00 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 10.02 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 10.02
Output dim: 7, lower bound: -1.9887453, upper bound: 1.9962101
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 10.02
Output dim: 7, lower bound: -1.9962097, upper bound: 1.9887454

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

Time for backsubstitution: 13.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5856
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5856

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9765336, upper bound: 1.9961678
time: 4.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9887031, upper bound: 1.9839899
time: 4.71 seconds

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

Time for backsubstitution: 14.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5856
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5856

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9839898, upper bound: 1.9887032
time: 5.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9961679, upper bound: 1.9765323
time: 6.05 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 26.08 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 26.08
Output dim: 7, lower bound: -1.9765336, upper bound: 1.9961678
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 26.08
Output dim: 7, lower bound: -1.9887031, upper bound: 1.9839899
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 26.08
Output dim: 7, lower bound: -1.9839898, upper bound: 1.9887032
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 26.08
Output dim: 7, lower bound: -1.9961679, upper bound: 1.9765323

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

Time for backsubstitution: 14.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5858

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9765306, upper bound: 1.9902767
time: 5.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9706369, upper bound: 1.9961667
time: 5.07 seconds

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

Time for backsubstitution: 13.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5858

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9887016, upper bound: 1.9780961
time: 4.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9828168, upper bound: 1.9839884
time: 5.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.5479779, 3.5700884
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.8666945, 2.8492246
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.8003278, 2.8187585
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.3906565, 3.3997684
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.3290973, 3.3368928
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.1132982, 3.1132982
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3615036, 2.3615036
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.5687590, 3.5784230

Time for backsubstitution: 14.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5858

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9839883, upper bound: 1.9828168
time: 6.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9780957, upper bound: 1.9887018
time: 4.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.5475440, 3.5703959
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.8661242, 2.8496218
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.8033767, 2.8144298
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.3920240, 3.3978310
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.3310428, 3.3341305
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.1132982, 3.1132982
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3615036, 2.3615036
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.5669241, 3.5797238

Time for backsubstitution: 14.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5858

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9961664, upper bound: 1.9706356
time: 6.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9902762, upper bound: 1.9765307
time: 7.35 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 28.40 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 28.40
Output dim: 7, lower bound: -1.9765306, upper bound: 1.9902767
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 28.40
Output dim: 7, lower bound: -1.9706369, upper bound: 1.9961667
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 28.40
Output dim: 7, lower bound: -1.9887016, upper bound: 1.9780961
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 28.40
Output dim: 7, lower bound: -1.9828168, upper bound: 1.9839884
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 28.40
Output dim: 7, lower bound: -1.9839883, upper bound: 1.9828168
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 28.40
Output dim: 7, lower bound: -1.9780957, upper bound: 1.9887018
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 28.40
Output dim: 7, lower bound: -1.9961664, upper bound: 1.9706356
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 28.40
Output dim: 7, lower bound: -1.9902762, upper bound: 1.9765307

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.5774188, 3.5570111
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.8521237, 2.8594420
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.8200831, 2.8066812
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.3536921, 3.3302908
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.3283353, 3.3236821
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.1132982, 3.1132982
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3615036, 2.3541574
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.5542288, 3.5515685

Time for backsubstitution: 14.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 4666

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9764142, upper bound: 1.9902618
time: 5.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9765161, upper bound: 1.9902460
time: 4.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.5799880, 3.5544367
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.8431115, 2.8684471
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.8164573, 2.8102961
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.3355274, 3.3484511
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.3259540, 3.3260617
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.1132982, 3.1132982
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3615036, 2.3615036
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.5648947, 3.5408926

Time for backsubstitution: 14.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 4666

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9705626, upper bound: 1.9961520
time: 4.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9706212, upper bound: 1.9960967
time: 5.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.5769820, 3.5573187
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.8515553, 2.8598385
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.8231330, 2.8023553
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.3550587, 3.3283539
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.3302817, 3.3209205
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.1132982, 3.1132982
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3615036, 2.3545783
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.5523920, 3.5528698

Time for backsubstitution: 14.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 4666

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9886278, upper bound: 1.9780813
time: 4.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9886870, upper bound: 1.9780317
time: 4.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.5795512, 3.5547447
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.8425431, 2.8688436
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.8195081, 2.8059702
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.3368950, 3.3465133
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.3279014, 3.3232996
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.1132982, 3.1132982
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3615036, 2.3615036
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.5630569, 3.5421939

Time for backsubstitution: 14.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 4666

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9827763, upper bound: 1.9839741
time: 4.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9828024, upper bound: 1.9838825
time: 5.11 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.5548701, 3.5795507
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.8690162, 2.8425424
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.8059707, 2.8207865
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.3465128, 3.3374653
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.3232999, 3.3287168
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.1132982, 3.1132982
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3615036, 2.3615036
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.5427275, 3.5630569

Time for backsubstitution: 14.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 4666

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9838824, upper bound: 1.9828022
time: 4.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9839740, upper bound: 1.9827765
time: 7.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.5574450, 3.5769811
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.8600116, 2.8515542
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.8023553, 2.8244128
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.3283529, 3.3556309
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.3209205, 3.3310976
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.1132982, 3.1132982
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3547544, 2.3615036
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.5534048, 3.5523915

Time for backsubstitution: 14.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 4666

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9780316, upper bound: 1.9886867
time: 4.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9780813, upper bound: 1.9886279
time: 6.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.5544372, 3.5798583
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.8684468, 2.8429389
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.8090205, 2.8164577
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.3478794, 3.3355279
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.3252463, 3.3259544
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.1132982, 3.1132982
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3615036, 2.3615036
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.5408926, 3.5643578

Time for backsubstitution: 14.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 4666

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9960965, upper bound: 1.9706212
time: 4.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9961521, upper bound: 1.9705628
time: 5.08 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.5570121, 3.5772891
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.8594422, 2.8519506
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.8054061, 2.8200831
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.3297195, 3.3536925
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.3228660, 3.3283353
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.1132982, 3.1132982
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3541574, 2.3615036
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.5515680, 3.5536928

Time for backsubstitution: 14.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 4666

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9902458, upper bound: 1.9765160
time: 4.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9902620, upper bound: 1.9764142
time: 8.45 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 27.69 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.69
Output dim: 7, lower bound: -1.9764142, upper bound: 1.9902618
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.69
Output dim: 7, lower bound: -1.9765161, upper bound: 1.9902460
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.69
Output dim: 7, lower bound: -1.9705626, upper bound: 1.9961520
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.69
Output dim: 7, lower bound: -1.9706212, upper bound: 1.9960967
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.69
Output dim: 7, lower bound: -1.9886278, upper bound: 1.9780813
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.69
Output dim: 7, lower bound: -1.9886870, upper bound: 1.9780317
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.69
Output dim: 7, lower bound: -1.9827763, upper bound: 1.9839741
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.69
Output dim: 7, lower bound: -1.9828024, upper bound: 1.9838825
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.69
Output dim: 7, lower bound: -1.9838824, upper bound: 1.9828022
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.69
Output dim: 7, lower bound: -1.9839740, upper bound: 1.9827765
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.69
Output dim: 7, lower bound: -1.9780316, upper bound: 1.9886867
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.69
Output dim: 7, lower bound: -1.9780813, upper bound: 1.9886279
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.69
Output dim: 7, lower bound: -1.9960965, upper bound: 1.9706212
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.69
Output dim: 7, lower bound: -1.9961521, upper bound: 1.9705628
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.69
Output dim: 7, lower bound: -1.9902458, upper bound: 1.9765160
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.69
Output dim: 7, lower bound: -1.9902620, upper bound: 1.9764142

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.5740161, 3.5574245
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.8536272, 2.8473864
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.8141522, 2.8074055
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.3468904, 3.3311181
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.3216491, 3.3244998
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.1132982, 3.1132982
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3615036, 2.3485386
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.5518942, 3.5518560

Time for backsubstitution: 14.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 4682

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9759812, upper bound: 1.9902603
time: 4.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9764128, upper bound: 1.9898553
time: 4.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.5774188, 3.5536098
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.8400679, 2.8594420
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.8200831, 2.8007503
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.3536921, 3.3234887
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.3283353, 3.3169959
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.1132982, 3.1132982
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3615036, 2.3541574
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.5542288, 3.5492339

Time for backsubstitution: 14.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 4682

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9760831, upper bound: 1.9902446
time: 4.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9765146, upper bound: 1.9898114
time: 5.22 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.5765853, 3.5548506
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.8446150, 2.8563914
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.8105273, 2.8110204
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.3287258, 3.3492770
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.3192687, 3.3268793
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.1132982, 3.1132982
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3615036, 2.3568423
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.5625601, 3.5411797

Time for backsubstitution: 14.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 4682

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9701293, upper bound: 1.9961504
time: 4.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9705611, upper bound: 1.9957208
time: 4.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.5799880, 3.5510359
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.8310556, 2.8684471
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.8164573, 2.8043652
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.3355274, 3.3416491
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.3259540, 3.3193755
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.1132982, 3.1132982
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3615036, 2.3615036
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.5648947, 3.5385580

Time for backsubstitution: 14.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 4682

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9702078, upper bound: 1.9960956
time: 4.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9706197, upper bound: 1.9956623
time: 5.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.5735793, 3.5577326
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.8530579, 2.8477852
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.8172021, 2.8030791
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.3482571, 3.3291812
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.3235946, 3.3217378
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.1132982, 3.1132982
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3615036, 2.3489590
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.5500574, 3.5531569

Time for backsubstitution: 14.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 4682

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9881948, upper bound: 1.9780797
time: 4.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9886263, upper bound: 1.9776748
time: 4.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.5769820, 3.5539179
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.8394995, 2.8598385
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.8231330, 2.7964244
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.3550587, 3.3215518
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.3302817, 3.3142343
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.1132982, 3.1132982
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3615036, 2.3545783
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.5523920, 3.5505352

Time for backsubstitution: 14.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 4682

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9882540, upper bound: 1.9780304
time: 4.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9886855, upper bound: 1.9775972
time: 4.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.5761504, 3.5551581
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.8440466, 2.8567903
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.8135762, 2.8066936
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.3300915, 3.3473392
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.3212142, 3.3241169
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.1132982, 3.1132982
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3615036, 2.3572626
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.5607224, 3.5424809

Time for backsubstitution: 14.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 4682

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9823429, upper bound: 1.9839724
time: 4.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9827747, upper bound: 1.9835498
time: 4.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.5795512, 3.5513434
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.8304873, 2.8688436
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.8195081, 2.8000393
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.3368950, 3.3397112
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.3279014, 3.3166134
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.1132982, 3.1132982
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3615036, 2.3615036
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.5630569, 3.5398593

Time for backsubstitution: 14.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 4682

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9823878, upper bound: 1.9838811
time: 5.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9828010, upper bound: 1.9834482
time: 5.19 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.5514693, 3.5799603
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.8705158, 2.8304868
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.8000388, 2.8215079
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.3397112, 3.3382921
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.3166137, 3.3295350
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.1132982, 3.1132982
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3615036, 2.3615036
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.5403929, 3.5633416

Time for backsubstitution: 14.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 4682

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9834481, upper bound: 1.9828005
time: 4.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9838810, upper bound: 1.9823873
time: 5.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.5548701, 3.5761495
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.8569603, 2.8425424
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.8059707, 2.8148556
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.3465128, 3.3306632
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.3232999, 3.3220305
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.1132982, 3.1132982
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3574383, 2.3615036
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.5427275, 3.5607224

Time for backsubstitution: 14.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 4682

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9835496, upper bound: 1.9827750
time: 4.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9839727, upper bound: 1.9823430
time: 5.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.5540423, 3.5773911
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.8615112, 2.8394985
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.7964244, 2.8251343
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.3215513, 3.3564568
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.3142333, 3.3319159
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.1132982, 3.1132982
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3554385, 2.3615036
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.5510702, 3.5526762

Time for backsubstitution: 14.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 4682

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9775968, upper bound: 1.9886851
time: 5.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9780302, upper bound: 1.9882536
time: 4.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.5574450, 3.5735803
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.8479557, 2.8515542
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.8023553, 2.8184819
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.3283529, 3.3488293
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.3209205, 3.3244114
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.1132982, 3.1132982
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3491347, 2.3615036
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.5534048, 3.5500574

Time for backsubstitution: 14.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 4682

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9776751, upper bound: 1.9886267
time: 4.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9780800, upper bound: 1.9881949
time: 5.45 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.5510364, 3.5802679
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.8699465, 2.8308856
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.8030887, 2.8171806
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.3410778, 3.3363547
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.3185592, 3.3267727
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.1132982, 3.1132982
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3615036, 2.3615036
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.5385580, 3.5646429

Time for backsubstitution: 14.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 4682

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9956622, upper bound: 1.9706197
time: 4.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9960951, upper bound: 1.9702076
time: 4.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.5544372, 3.5764575
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.8563910, 2.8429389
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.8090205, 2.8105268
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.3478794, 3.3287258
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.3252463, 3.3192682
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.1132982, 3.1132982
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3568423, 2.3615036
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.5408926, 3.5620232

Time for backsubstitution: 14.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 4682

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9957212, upper bound: 1.9705612
time: 4.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9961508, upper bound: 1.9701291
time: 5.22 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.5536094, 3.5776987
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.8609419, 2.8398974
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.7994742, 2.8208060
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.3229179, 3.3545184
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.3161807, 3.3291535
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.1132982, 3.1132982
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3548424, 2.3615036
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.5492334, 3.5539770

Time for backsubstitution: 14.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 4682

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9898110, upper bound: 1.9765145
time: 5.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9902444, upper bound: 1.9760830
time: 4.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.5570121, 3.5738878
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.8473864, 2.8519506
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.8054061, 2.8141522
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.3297195, 3.3468904
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.3228660, 3.3216491
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.1132982, 3.1132982
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3485386, 2.3615036
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.5515680, 3.5513582

Time for backsubstitution: 14.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 4682

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9898558, upper bound: 1.9764129
time: 5.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9902606, upper bound: 1.9759814
time: 5.32 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 24.86 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.86
Output dim: 7, lower bound: -1.9759812, upper bound: 1.9902603
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.86
Output dim: 7, lower bound: -1.9764128, upper bound: 1.9898553
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.86
Output dim: 7, lower bound: -1.9760831, upper bound: 1.9902446
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.86
Output dim: 7, lower bound: -1.9765146, upper bound: 1.9898114
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.86
Output dim: 7, lower bound: -1.9701293, upper bound: 1.9961504
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.86
Output dim: 7, lower bound: -1.9705611, upper bound: 1.9957208
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.86
Output dim: 7, lower bound: -1.9702078, upper bound: 1.9960956
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.86
Output dim: 7, lower bound: -1.9706197, upper bound: 1.9956623
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.86
Output dim: 7, lower bound: -1.9881948, upper bound: 1.9780797
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.86
Output dim: 7, lower bound: -1.9886263, upper bound: 1.9776748
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.86
Output dim: 7, lower bound: -1.9882540, upper bound: 1.9780304
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.86
Output dim: 7, lower bound: -1.9886855, upper bound: 1.9775972
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.86
Output dim: 7, lower bound: -1.9823429, upper bound: 1.9839724
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.86
Output dim: 7, lower bound: -1.9827747, upper bound: 1.9835498
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.86
Output dim: 7, lower bound: -1.9823878, upper bound: 1.9838811
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.86
Output dim: 7, lower bound: -1.9828010, upper bound: 1.9834482
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.86
Output dim: 7, lower bound: -1.9834481, upper bound: 1.9828005
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.86
Output dim: 7, lower bound: -1.9838810, upper bound: 1.9823873
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.86
Output dim: 7, lower bound: -1.9835496, upper bound: 1.9827750
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.86
Output dim: 7, lower bound: -1.9839727, upper bound: 1.9823430
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.86
Output dim: 7, lower bound: -1.9775968, upper bound: 1.9886851
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.86
Output dim: 7, lower bound: -1.9780302, upper bound: 1.9882536
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.86
Output dim: 7, lower bound: -1.9776751, upper bound: 1.9886267
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.86
Output dim: 7, lower bound: -1.9780800, upper bound: 1.9881949
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.86
Output dim: 7, lower bound: -1.9956622, upper bound: 1.9706197
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.86
Output dim: 7, lower bound: -1.9960951, upper bound: 1.9702076
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.86
Output dim: 7, lower bound: -1.9957212, upper bound: 1.9705612
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.86
Output dim: 7, lower bound: -1.9961508, upper bound: 1.9701291
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.86
Output dim: 7, lower bound: -1.9898110, upper bound: 1.9765145
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.86
Output dim: 7, lower bound: -1.9902444, upper bound: 1.9760830
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.86
Output dim: 7, lower bound: -1.9898558, upper bound: 1.9764129
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.86
Output dim: 7, lower bound: -1.9902606, upper bound: 1.9759814

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.5700660, 3.5518456
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.8758903, 2.8641770
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.8142262, 2.8073630
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.3263597, 3.3021288
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9762487, 3.9762487
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.3271155, 3.3317442
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.1132982, 3.1132982
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.3615036, 2.3403978
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.5376863, 3.5417943

Time for backsubstitution: 13.98 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=3.113298177719116
rel_dist={7: [-1.996222388869347, 1.996221151991417]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 5856
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 565

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7159505, upper bound: 1.7205214
time: 5.16 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7205205, upper bound: 1.7159499
time: 5.50 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 10.85 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 10.85
Output dim: 7, lower bound: -1.7159505, upper bound: 1.7205214
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 10.85
Output dim: 7, lower bound: -1.7205205, upper bound: 1.7159499

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.4007006, 3.3878183
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6900578, 2.6997147
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6785636, 2.6705048
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.1337996, 3.1297007
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9595900, 3.9658308
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.1254506, 3.1225743
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0526352, 3.0552068
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.2674446, 2.2491202
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3861904, 3.3796196

Time for backsubstitution: 13.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5856
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5856

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7057160, upper bound: 1.7205009
time: 5.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7159283, upper bound: 1.7102843
time: 5.53 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.3878183, 3.4007010
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6997147, 2.6900578
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6705050, 2.6785634
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.1297007, 3.1338005
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9658308, 3.9595895
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.1225743, 3.1254511
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0552063, 3.0526347
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.2491207, 2.2674444
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3796196, 3.3861909

Time for backsubstitution: 14.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5856
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5856

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7102835, upper bound: 1.7159283
time: 5.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7204999, upper bound: 1.7057153
time: 5.28 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 24.87 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.87
Output dim: 7, lower bound: -1.7057160, upper bound: 1.7205009
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.87
Output dim: 7, lower bound: -1.7159283, upper bound: 1.7102843
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.87
Output dim: 7, lower bound: -1.7102835, upper bound: 1.7159283
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.87
Output dim: 7, lower bound: -1.7204999, upper bound: 1.7057153

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.4006429, 3.3875108
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6899862, 2.6993175
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6755137, 2.6699266
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.1324329, 3.1294408
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9567213, 3.9652882
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.1235051, 3.1222067
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0500288, 3.0547099
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.2673650, 2.2487001
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3859396, 3.3783188

Time for backsubstitution: 13.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5858

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7057138, upper bound: 1.7176045
time: 5.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7028817, upper bound: 1.7204997
time: 5.25 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.4003930, 3.3877587
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6896620, 2.6996429
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6779866, 2.6674552
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.1335402, 3.1283336
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9590464, 3.9629636
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.1250834, 3.1206288
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0521383, 3.0526009
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.2670245, 2.2490408
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3848906, 3.3793669

Time for backsubstitution: 14.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5858

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7159277, upper bound: 1.7073816
time: 8.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7131018, upper bound: 1.7102831
time: 5.19 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.3877587, 3.4003935
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6996431, 2.6896605
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6674552, 2.6779871
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.1283331, 3.1335406
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9629641, 3.9590468
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.1206288, 3.1250837
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0526009, 3.0521379
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.2490411, 2.2670243
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3793669, 3.3848896

Time for backsubstitution: 14.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5858

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7028832, upper bound: 1.7131012
time: 7.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7028832, upper bound: 1.7159275
time: 4.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.3875108, 3.4006429
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6993189, 2.6899855
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6699262, 2.6755137
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.1294403, 3.1324334
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9652872, 3.9567218
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.1222072, 3.1235051
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0547094, 3.0500288
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.2487006, 2.2673647
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3783178, 3.3859391

Time for backsubstitution: 14.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5858

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7204994, upper bound: 1.7028823
time: 5.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7028831, upper bound: 1.7057154
time: 6.12 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 25.47 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.47
Output dim: 7, lower bound: -1.7057138, upper bound: 1.7176045
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.47
Output dim: 7, lower bound: -1.7028817, upper bound: 1.7204997
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.47
Output dim: 7, lower bound: -1.7159277, upper bound: 1.7073816
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.47
Output dim: 7, lower bound: -1.7131018, upper bound: 1.7102831
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.47
Output dim: 7, lower bound: -1.7028832, upper bound: 1.7131012
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.47
Output dim: 7, lower bound: -1.7028832, upper bound: 1.7159275
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.47
Output dim: 7, lower bound: -1.7204994, upper bound: 1.7028823
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.47
Output dim: 7, lower bound: -1.7028831, upper bound: 1.7057154

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.4075370, 3.3958745
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6884542, 2.6926353
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6796126, 2.6719551
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.0805101, 3.0671377
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9566479, 3.9652262
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1073217, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.1166892, 3.1140306
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0496931, 3.0546308
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.2435942, 2.2201703
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3599081, 3.3583879

Time for backsubstitution: 14.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 4666

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7051039, upper bound: 1.7175948
time: 5.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7057050, upper bound: 1.7169224
time: 5.06 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.4090037, 3.3944035
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6833043, 2.6977811
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6775413, 2.6740208
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.0701303, 3.0775146
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9566593, 3.9652138
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1056452, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.1153283, 3.1153903
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0499487, 3.0543747
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.2388353, 2.2249153
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3660030, 3.3522873

Time for backsubstitution: 14.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 4666

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7022136, upper bound: 1.7204900
time: 5.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7028731, upper bound: 1.7198633
time: 6.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.4072871, 3.3961225
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6881280, 2.6929607
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6820874, 2.6694832
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.0816174, 3.0660305
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9589729, 3.9629016
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1098566, 3.1136003
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.1182675, 3.1124525
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0518017, 3.0525217
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.2432537, 2.2205110
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3588591, 3.3594370

Time for backsubstitution: 14.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 4666

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7153179, upper bound: 1.7073722
time: 5.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7159190, upper bound: 1.7066900
time: 12.48 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.4087558, 3.3946514
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6829791, 2.6981061
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6800151, 2.6715488
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.0712366, 3.0764074
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9589844, 3.9628892
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1081800, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.1169076, 3.1138120
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0520573, 3.0522656
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.2384949, 2.2252560
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3649530, 3.3533354

Time for backsubstitution: 14.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 4666

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7124334, upper bound: 1.7102741
time: 5.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7130931, upper bound: 1.7096426
time: 5.47 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.3946528, 3.4087543
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6981063, 2.6829784
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6715484, 2.6800151
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.0764074, 3.0712376
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9628887, 3.9589844
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1081805
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.1138120, 3.1169076
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0522652, 3.0520582
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.2252560, 2.2384944
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3533354, 3.3649526

Time for backsubstitution: 14.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 4666

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7096427, upper bound: 1.7130931
time: 5.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7102742, upper bound: 1.7124334
time: 5.26 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.3961234, 3.4072862
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6929612, 2.6881280
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6694827, 2.6820874
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.0660305, 3.0816178
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9629021, 3.9589725
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1136007, 3.1098561
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.1124520, 3.1182680
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0525217, 3.0518026
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.2205114, 2.2432530
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3594370, 3.3588581

Time for backsubstitution: 14.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 4666

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7066893, upper bound: 1.7159189
time: 5.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7073723, upper bound: 1.7153175
time: 5.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.3944049, 3.4090037
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6977811, 2.6833034
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6740203, 2.6775417
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.0775146, 3.0701303
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9652138, 3.9566598
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1056457
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.1153903, 3.1153293
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0543747, 3.0499492
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.2249155, 2.2388349
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3522873, 3.3660030

Time for backsubstitution: 14.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 4666

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7198631, upper bound: 1.7028732
time: 5.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7204906, upper bound: 1.7022121
time: 5.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.3958755, 3.4075356
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6926360, 2.6884532
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6719546, 2.6796131
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.0671377, 3.0805101
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9652271, 3.9566474
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1073213
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.1140304, 3.1166897
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0546303, 3.0496936
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.2201710, 2.2435935
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3583879, 3.3599076

Time for backsubstitution: 14.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 4666

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7066907, upper bound: 1.7057047
time: 5.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7175951, upper bound: 1.7051037
time: 5.47 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 25.24 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.24
Output dim: 7, lower bound: -1.7051039, upper bound: 1.7175948
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.24
Output dim: 7, lower bound: -1.7057050, upper bound: 1.7169224
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.24
Output dim: 7, lower bound: -1.7022136, upper bound: 1.7204900
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.24
Output dim: 7, lower bound: -1.7028731, upper bound: 1.7198633
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.24
Output dim: 7, lower bound: -1.7153179, upper bound: 1.7073722
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.24
Output dim: 7, lower bound: -1.7159190, upper bound: 1.7066900
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.24
Output dim: 7, lower bound: -1.7124334, upper bound: 1.7102741
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.24
Output dim: 7, lower bound: -1.7130931, upper bound: 1.7096426
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.24
Output dim: 7, lower bound: -1.7096427, upper bound: 1.7130931
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.24
Output dim: 7, lower bound: -1.7102742, upper bound: 1.7124334
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.24
Output dim: 7, lower bound: -1.7066893, upper bound: 1.7159189
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.24
Output dim: 7, lower bound: -1.7073723, upper bound: 1.7153175
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.24
Output dim: 7, lower bound: -1.7198631, upper bound: 1.7028732
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.24
Output dim: 7, lower bound: -1.7204906, upper bound: 1.7022121
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.24
Output dim: 7, lower bound: -1.7066907, upper bound: 1.7057047
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.24
Output dim: 7, lower bound: -1.7175951, upper bound: 1.7051037

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.4041343, 3.3946533
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6841459, 2.6805797
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6736827, 2.6698270
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.0737085, 3.0646954
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9566345, 3.9651866
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1054506, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.1100039, 3.1116323
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0486016, 3.0515733
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.2415774, 2.2145514
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3575735, 3.3575516

Time for backsubstitution: 14.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 4682

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7041618, upper bound: 1.7175934
time: 5.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7051015, upper bound: 1.7168640
time: 5.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.4063125, 3.3924737
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6763983, 2.6883256
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6774840, 2.6660237
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.0780668, 3.0603356
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9566078, 3.9652119
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1066484, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.1142917, 3.1073444
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0466361, 3.0535388
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.2379744, 2.2181535
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3590708, 3.3560534

Time for backsubstitution: 14.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 4682

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7014776, upper bound: 1.7169201
time: 5.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7022110, upper bound: 1.7162415
time: 5.17 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.4056029, 3.3931823
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6789961, 2.6857255
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6716113, 2.6718926
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.0633287, 3.0750713
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9566460, 3.9651742
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1037760, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.1086421, 3.1129920
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0488582, 3.0513172
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.2368195, 2.2192965
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3636684, 3.3514509

Time for backsubstitution: 14.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 4682

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7014776, upper bound: 1.7204893
time: 6.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7022110, upper bound: 1.7195958
time: 5.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.4077792, 3.3910027
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6712484, 2.6934714
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6754127, 2.6680899
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.0676870, 3.0707130
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9566212, 3.9651995
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1049719, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.1129317, 3.1087041
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0468917, 3.0532823
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.2332165, 2.2228982
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3651657, 3.3499527

Time for backsubstitution: 14.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 4682

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7014776, upper bound: 1.7198610
time: 5.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7028706, upper bound: 1.7190648
time: 5.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.4038844, 3.3949018
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6838207, 2.6809051
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6761565, 2.6673546
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.0748158, 3.0635881
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9589577, 3.9628620
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1079855, 3.1129265
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.1115813, 3.1100540
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0507102, 3.0494647
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.2412369, 2.2148921
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3565245, 3.3586006

Time for backsubstitution: 14.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 4682

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7143598, upper bound: 1.7073706
time: 5.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7153155, upper bound: 1.7066482
time: 7.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.4060626, 3.3927212
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6760721, 2.6886508
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6799579, 2.6635523
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.0791750, 3.0592289
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9589329, 3.9628873
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1091833, 3.1117296
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.1158690, 3.1057663
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0487456, 3.0514297
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.2376339, 2.2184942
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3580198, 3.3571024

Time for backsubstitution: 13.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 4682

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7147649, upper bound: 1.7066877
time: 5.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7159165, upper bound: 1.7060222
time: 5.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.4053531, 3.3934307
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6786709, 2.6860504
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6740842, 2.6694202
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.0644350, 3.0739641
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9589710, 3.9628496
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1063108, 3.1138687
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.1102214, 3.1114135
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0509658, 3.0492082
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.2364790, 2.2196372
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3626184, 3.3524995

Time for backsubstitution: 13.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 4682

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7014775, upper bound: 1.7102713
time: 4.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7124310, upper bound: 1.7093953
time: 5.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.4075313, 3.3912501
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6709232, 2.6937964
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6778855, 2.6656179
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.0687943, 3.0696058
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9589443, 3.9628749
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1075068, 3.1134048
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.1145091, 3.1071258
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0490012, 3.0511737
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.2328761, 2.2232392
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3641148, 3.3510013

Time for backsubstitution: 14.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 4682

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7122806, upper bound: 1.7096412
time: 6.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7130906, upper bound: 1.7088488
time: 7.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.3912501, 3.4075308
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6937971, 2.6709228
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6656184, 2.6778855
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.0696058, 3.0687943
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9628754, 3.9589448
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1134052, 3.1075068
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.1071258, 3.1145096
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0511737, 3.0490007
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.2232392, 2.2328756
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3510008, 3.3641148

Time for backsubstitution: 14.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 4682

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7088488, upper bound: 1.7130911
time: 5.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7096406, upper bound: 1.7122805
time: 5.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.3934302, 3.4053535
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6860504, 2.6786709
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6694198, 2.6740842
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.0739641, 3.0644350
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9628487, 3.9589705
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1063099
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.1114135, 3.1102214
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0492082, 3.0509663
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.2196372, 2.2364786
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3525000, 3.3626184

Time for backsubstitution: 14.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 4682

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7093956, upper bound: 1.7124315
time: 5.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7102719, upper bound: 1.7116902
time: 4.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.3927207, 3.4060626
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6886511, 2.6760724
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6635528, 2.6799579
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.0592289, 3.0791745
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9628868, 3.9589329
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1117296, 3.1091824
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.1057658, 3.1158700
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0514293, 3.0487452
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.2184947, 2.2376342
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3571024, 3.3580203

Time for backsubstitution: 13.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 4682

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7060230, upper bound: 1.7159167
time: 5.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7066871, upper bound: 1.7147647
time: 5.12 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.3949027, 3.4038854
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6809053, 2.6838205
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6673541, 2.6761565
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.0635881, 3.0748158
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9628620, 3.9589586
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1129274, 3.1079855
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.1100535, 3.1115818
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0494647, 3.0507107
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.2148926, 2.2412372
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3586006, 3.3565240

Time for backsubstitution: 13.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 4682

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7066472, upper bound: 1.7153152
time: 5.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7066884, upper bound: 1.7143590
time: 5.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.3910022, 3.4077802
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6934719, 2.6712477
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6680903, 2.6754131
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.0707130, 3.0676875
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9652004, 3.9566202
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1049719
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.1087031, 3.1129313
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0532823, 3.0468917
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.2228987, 2.2332160
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3499527, 3.3651657

Time for backsubstitution: 13.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 4682

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7190643, upper bound: 1.7028711
time: 5.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7198610, upper bound: 1.7020684
time: 5.26 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.3931823, 3.4056025
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6857252, 2.6789958
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6718926, 2.6716108
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.0750713, 3.0633283
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9651737, 3.9566460
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1037750
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.1129909, 3.1086431
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0513167, 3.0488577
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.2192967, 2.2368190
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3514509, 3.3636684

Time for backsubstitution: 13.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 4682

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7195963, upper bound: 1.7022102
time: 5.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7204883, upper bound: 1.7014768
time: 5.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.3924727, 3.4063120
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6883259, 2.6763976
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6660237, 2.6774845
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.0603361, 3.0780668
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9652119, 3.9566078
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1066475
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.1073432, 3.1142917
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0535388, 3.0466361
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.2181532, 2.2379746
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3560534, 3.3590703

Time for backsubstitution: 13.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 4682

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7162421, upper bound: 1.7057026
time: 5.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7169196, upper bound: 1.7045702
time: 5.24 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.3946528, 3.4041343
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6805801, 2.6841455
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6698270, 2.6736822
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.0646954, 3.0737081
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9651871, 3.9566336
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.1138687, 3.1054506
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.1116328, 3.1100035
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0515733, 3.0486016
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.2145512, 2.2415776
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3575516, 3.3575735

Time for backsubstitution: 13.72 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=3.0680713653564453
rel_dist={7: [-1.7205280069910254, 1.7205289191635718]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 5856
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 565

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6159423, upper bound: 1.6196668
time: 4.93 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6196670, upper bound: 1.6159419
time: 5.19 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 10.31 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 10.31
Output dim: 7, lower bound: -1.6159423, upper bound: 1.6196668
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 10.31
Output dim: 7, lower bound: -1.6196670, upper bound: 1.6159419

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.3441362, 3.3344741
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6368704, 2.6441126
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6322579, 2.6262138
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.0453348, 3.0422602
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.8983431, 3.9030237
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0617542, 3.0677204
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.0552430, 3.0530851
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0145607, 3.0164895
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.2182012, 2.2044578
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3216801, 3.3167510

Time for backsubstitution: 13.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5856
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5856

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6082911, upper bound: 1.6196499
time: 5.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.6082911, upper bound: 1.6120121
time: 5.11 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.3344736, 3.3441358
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6441135, 2.6368699
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6262136, 2.6322577
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.0422602, 3.0453348
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9030237, 3.8983426
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0677204, 3.0617542
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.0530858, 3.0552425
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0164890, 3.0145602
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.2044582, 2.2182009
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3167515, 3.3216796

Time for backsubstitution: 13.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5856
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5856

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.6120121, upper bound: 1.6159247
time: 5.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6196498, upper bound: 1.6082892
time: 5.02 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 24.36 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.36
Output dim: 7, lower bound: -1.6082911, upper bound: 1.6196499
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 24.36
Output dim: 7, lower bound: -1.6082911, upper bound: 1.6120121
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 24.36
Output dim: 7, lower bound: -1.6120121, upper bound: 1.6159247
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.36
Output dim: 7, lower bound: -1.6196498, upper bound: 1.6082892

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.3440156, 3.3341665
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6367178, 2.6437154
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6292081, 2.6250181
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.0439672, 3.0417228
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.8954744, 3.9018998
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0586233, 3.0664907
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.0532966, 3.0523229
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0119543, 3.0154653
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.2180362, 2.2040377
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3211660, 3.3154507

Time for backsubstitution: 13.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5858

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.6067104, upper bound: 1.6178677
time: 4.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6067104, upper bound: 1.6196492
time: 4.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.3341660, 3.3440156
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6437159, 2.6367164
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6250176, 2.6292081
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -3.0417223, 3.0439677
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9019003, 3.8954754
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0664911, 3.0586228
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.0523229, 3.0532968
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0154657, 3.0119543
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.2040381, 2.2180362
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.3154497, 3.3211660

Time for backsubstitution: 13.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5858
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5858

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6196496, upper bound: 1.6067082
time: 4.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.6178679, upper bound: 1.6082900
time: 5.69 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 24.44 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 24.44
Output dim: 7, lower bound: -1.6067104, upper bound: 1.6178677
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.44
Output dim: 7, lower bound: -1.6067104, upper bound: 1.6196492
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.44
Output dim: 7, lower bound: -1.6196496, upper bound: 1.6067082
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 24.44
Output dim: 7, lower bound: -1.6178679, upper bound: 1.6082900

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.3520103, 3.3410592
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6300349, 2.6408925
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6312366, 2.6285954
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -2.9816637, 2.9872026
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.8954105, 3.9018254
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0485830, 3.0577073
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.0451207, 3.0451665
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0118103, 3.0151300
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.1895065, 2.1790667
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.2997055, 3.2894192

Time for backsubstitution: 13.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 4666

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6063040, upper bound: 1.6196400
time: 5.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6066987, upper bound: 1.6193392
time: 5.25 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.3410602, 3.3520093
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6408925, 2.6300342
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6285949, 2.6312361
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -2.9872026, 2.9816647
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9018250, 3.8954101
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0577078, 3.0485835
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.0451665, 3.0451207
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0151291, 3.0118108
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.1790667, 2.1895063
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.2894192, 3.2997055

Time for backsubstitution: 13.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4666
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 4666

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6193398, upper bound: 1.6066988
time: 5.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6196392, upper bound: 1.6063019
time: 5.07 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 24.18 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.18
Output dim: 7, lower bound: -1.6063040, upper bound: 1.6196400
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.18
Output dim: 7, lower bound: -1.6066987, upper bound: 1.6193392
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.18
Output dim: 7, lower bound: -1.6193398, upper bound: 1.6066988
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.18
Output dim: 7, lower bound: -1.6196392, upper bound: 1.6063019

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.3486075, 3.3392930
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6237898, 2.6288369
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6253047, 2.6255169
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -2.9748621, 2.9836698
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.8953896, 3.9017859
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0467138, 3.0567346
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.0384345, 3.0416963
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0102277, 3.0120726
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.1865895, 2.1734478
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.2973709, 3.2882080

Time for backsubstitution: 13.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 4682

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6051906, upper bound: 1.6196359
time: 5.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.6051906, upper bound: 1.6183773
time: 4.50 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.3502421, 3.3376584
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6179790, 2.6346462
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6281571, 2.6226645
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -2.9781313, 2.9804010
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.8953705, 3.9018049
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0476112, 3.0558367
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.0416503, 3.0384803
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0087533, 3.0135465
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.1838877, 2.1761491
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.2984943, 3.2870846

Time for backsubstitution: 13.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 4682

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6051906, upper bound: 1.6193351
time: 4.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.6051906, upper bound: 1.6180982
time: 5.16 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.3376575, 3.3502412
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6346464, 2.6179786
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6226649, 2.6281571
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -2.9804010, 2.9781318
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9018059, 3.8953705
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0558367, 3.0476108
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.0384803, 3.0416508
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0135465, 3.0087533
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.1761496, 2.1838875
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.2870846, 3.2984943

Time for backsubstitution: 13.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 4682

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.6089068, upper bound: 1.6066948
time: 5.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6193367, upper bound: 1.6054720
time: 5.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.3392940, 3.3486080
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6288366, 2.6237895
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6255164, 2.6253052
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -2.9836702, 2.9748621
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.9017868, 3.8953896
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0567350, 3.0467129
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.0416961, 3.0384345
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0120730, 3.0102277
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.1734478, 2.1865897
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.2882080, 3.2973709

Time for backsubstitution: 13.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4682
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 4682

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.6089068, upper bound: 1.6062993
time: 5.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.6096786, upper bound: 1.6051880
time: 5.15 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 24.38 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.38
Output dim: 7, lower bound: -1.6051906, upper bound: 1.6196359
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.38
Output dim: 7, lower bound: -1.6051906, upper bound: 1.6183773
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.38
Output dim: 7, lower bound: -1.6051906, upper bound: 1.6193351
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.38
Output dim: 7, lower bound: -1.6051906, upper bound: 1.6180982
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.38
Output dim: 7, lower bound: -1.6089068, upper bound: 1.6066948
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.38
Output dim: 7, lower bound: -1.6193367, upper bound: 1.6054720
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.38
Output dim: 7, lower bound: -1.6089068, upper bound: 1.6062993
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.38
Output dim: 7, lower bound: -1.6096786, upper bound: 1.6051880

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.3437266, 3.3337140
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6429276, 2.6456275
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6253128, 2.6254745
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -2.9495020, 2.9546804
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.8871574, 3.8945832
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0390778, 3.0480070
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.0439010, 3.0479250
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0115676, 3.0132489
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.1794682, 2.1653070
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.2831640, 3.2757792

Time for backsubstitution: 13.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 6220

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6029614, upper bound: 1.6196298
time: 5.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.6051830, upper bound: 1.6174016
time: 5.46 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.3453612, 3.3320794
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6371160, 2.6514368
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6281643, 2.6226220
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -2.9527712, 2.9514117
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.8871384, 3.8946023
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0399742, 3.0471091
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.0471177, 3.0447090
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0100932, 3.0147228
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.1767664, 2.1680083
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.2842865, 3.2746558

Time for backsubstitution: 13.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6220

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6032292, upper bound: 1.6193305
time: 5.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.6054670, upper bound: 1.6171067
time: 5.21 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.3320804, 3.3453598
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6514373, 2.6371160
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6226215, 2.6281648
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -2.9514112, 2.9527707
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.8946018, 3.8871374
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0471096, 3.0399742
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.0447087, 3.0471177
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0147223, 3.0100932
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.1680088, 2.1767659
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.2746553, 3.2842870

Time for backsubstitution: 13.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 6220

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.6171071, upper bound: 1.6054663
time: 5.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6193308, upper bound: 1.6032290
time: 5.22 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 24.43 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 24.43
Output dim: 7, lower bound: -1.6029614, upper bound: 1.6196298
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 24.43
Output dim: 7, lower bound: -1.6051830, upper bound: 1.6174016
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 24.43
Output dim: 7, lower bound: -1.6032292, upper bound: 1.6193305
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 24.43
Output dim: 7, lower bound: -1.6054670, upper bound: 1.6171067
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 24.43
Output dim: 7, lower bound: -1.6171071, upper bound: 1.6054663
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 24.43
Output dim: 7, lower bound: -1.6193308, upper bound: 1.6032290

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.3357940, 3.3281951
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6326032, 2.6307547
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6234035, 2.6241341
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -2.9483528, 2.9538879
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.8825750, 3.8879743
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0323443, 3.0383306
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.0400672, 3.0452511
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0078731, 3.0106874
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.1602373, 2.1519589
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.2745209, 3.2697735

Time for backsubstitution: 13.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 4573

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.6010914, upper bound: 1.6177621
time: 5.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6010914, upper bound: 1.6196225
time: 4.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.3374286, 3.3265615
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6267962, 2.6365640
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6262550, 2.6212831
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -2.9516220, 2.9506192
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.8825560, 3.8879933
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0332408, 3.0374327
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.0432830, 3.0420377
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0063977, 3.0121617
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.1575346, 2.1546626
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.2756424, 3.2686520

Time for backsubstitution: 13.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 4573

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.6032212, upper bound: 1.6174657
time: 5.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6013187, upper bound: 1.6193226
time: 6.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.3265605, 3.3374281
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6365638, 2.6267960
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6212826, 2.6262555
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -2.9506197, 2.9516220
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.8879938, 3.8825555
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0374331, 3.0332398
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.0420375, 3.0432825
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0121617, 3.0063982
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.1546631, 2.1575346
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.2686520, 3.2756429

Time for backsubstitution: 13.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4573
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 4573

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6193230, upper bound: 1.6013181
time: 5.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.6174660, upper bound: 1.6032210
time: 5.96 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 25.24 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 25.24
Output dim: 7, lower bound: -1.6010914, upper bound: 1.6177621
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 25.24
Output dim: 7, lower bound: -1.6010914, upper bound: 1.6196225
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 25.24
Output dim: 7, lower bound: -1.6032212, upper bound: 1.6174657
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 25.24
Output dim: 7, lower bound: -1.6013187, upper bound: 1.6193226
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 25.24
Output dim: 7, lower bound: -1.6193230, upper bound: 1.6013181
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 25.24
Output dim: 7, lower bound: -1.6174660, upper bound: 1.6032210

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.3315630, 3.3244905
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6149721, 2.6158333
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6360950, 2.6349344
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -2.9581485, 2.9612718
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.8848591, 3.8906379
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0456214, 3.0538387
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.0517945, 3.0589497
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0158625, 3.0175266
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.1649199, 2.1574285
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.2812853, 3.2755623

Time for backsubstitution: 13.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 944

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.6005667, upper bound: 1.6172378
time: 6.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6005679, upper bound: 1.6195620
time: 5.08 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.3331966, 3.3228569
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6091652, 2.6216426
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6389465, 2.6320834
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -2.9614177, 2.9580030
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.8848400, 3.8906569
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0465178, 3.0529413
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.0550103, 3.0557361
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0143881, 3.0190010
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.1622176, 2.1601322
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.2824087, 3.2744408

Time for backsubstitution: 13.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 944

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.6008197, upper bound: 1.6167219
time: 5.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6008194, upper bound: 1.6192621
time: 5.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.3228569, 3.3331957
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6216431, 2.6091647
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6320839, 2.6389461
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -2.9580016, 2.9614177
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.8906574, 3.8848395
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0529418, 3.0465169
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.0557351, 3.0550113
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0190010, 3.0143876
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.1601319, 2.1622176
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.2744408, 3.2824078

Time for backsubstitution: 13.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 944

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6192623, upper bound: 1.6008187
time: 5.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.6167225, upper bound: 1.6008203
time: 5.36 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 24.74 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 24.74
Output dim: 7, lower bound: -1.6005667, upper bound: 1.6172378
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 24.74
Output dim: 7, lower bound: -1.6005679, upper bound: 1.6195620
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 24.74
Output dim: 7, lower bound: -1.6008197, upper bound: 1.6167219
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 24.74
Output dim: 7, lower bound: -1.6008194, upper bound: 1.6192621
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 24.74
Output dim: 7, lower bound: -1.6192623, upper bound: 1.6008187
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 24.74
Output dim: 7, lower bound: -1.6167225, upper bound: 1.6008203

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.3337860, 3.3263035
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6117568, 2.6136220
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6370931, 2.6361578
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -2.9393969, 2.9448662
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.8835258, 3.8891149
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0395207, 3.0485010
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.0522552, 3.0593252
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0148215, 3.0167785
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.1591303, 2.1523616
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.2783928, 3.2715321

Time for backsubstitution: 13.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 6140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.5990721, upper bound: 1.6180687
time: 5.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.5990705, upper bound: 1.6195601
time: 5.44 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.3354187, 3.3246698
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6059499, 2.6194305
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6399446, 2.6333067
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -2.9426651, 2.9415956
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.8835068, 3.8891339
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0404153, 3.0476036
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.0554719, 3.0561118
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0133471, 3.0182528
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.1564286, 2.1550653
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.2795153, 3.2704105

Time for backsubstitution: 13.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 6140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.6008168, upper bound: 1.6177681
time: 5.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.5993046, upper bound: 1.6192603
time: 5.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.3246708, 3.3354187
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.6194310, 2.6059494
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6333070, 2.6399453
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -2.9415960, 2.9426651
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.8891344, 3.8835063
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0476041, 3.0404158
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.0561118, 3.0554714
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0182528, 3.0133476
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.1550648, 2.1564283
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.2704105, 3.2795153

Time for backsubstitution: 13.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6140
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6192601, upper bound: 1.5993040
time: 5.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.6177683, upper bound: 1.6008171
time: 5.50 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 24.87 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 24.87
Output dim: 7, lower bound: -1.5990721, upper bound: 1.6180687
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 24.87
Output dim: 7, lower bound: -1.5990705, upper bound: 1.6195601
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 24.87
Output dim: 7, lower bound: -1.6008168, upper bound: 1.6177681
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 24.87
Output dim: 7, lower bound: -1.5993046, upper bound: 1.6192603
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 24.87
Output dim: 7, lower bound: -1.6192601, upper bound: 1.5993040
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 24.87
Output dim: 7, lower bound: -1.6177683, upper bound: 1.6008171

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.3002357, 3.2969446
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.5767603, 2.5829918
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6216497, 2.6185129
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -2.9518347, 2.9546618
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.8876610, 3.8939853
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0278602, 3.0342426
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.0670052, 3.0766973
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0209675, 3.0219955
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.1556649, 2.1493282
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.2792358, 3.2722502

Time for backsubstitution: 13.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5801

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.5976606, upper bound: 1.6181948
time: 5.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.5976606, upper bound: 1.6195556
time: 5.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.3018694, 3.2953110
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.5709534, 2.5888004
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6245012, 2.6156619
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -2.9551039, 2.9513917
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.8876419, 3.8940043
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0287557, 3.0333452
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.0702209, 3.0734835
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0194931, 3.0234699
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.1529632, 2.1520319
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.2803593, 3.2711287

Time for backsubstitution: 13.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5801

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.5993006, upper bound: 1.6178925
time: 5.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.5979006, upper bound: 1.6192554
time: 5.08 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.2953110, 3.3018689
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.5888014, 2.5709527
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.6156626, 2.6245015
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -2.9513922, 2.9551034
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.8940048, 3.8876419
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0333457, 3.0287557
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.0734835, 3.0702209
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0234699, 3.0194931
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.1520314, 2.1529634
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.2711287, 3.2803588

Time for backsubstitution: 13.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5801
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5801

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.6042646, upper bound: 1.5978989
time: 5.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.6042646, upper bound: 1.5993005
time: 7.19 seconds

## Summary of splitting (split count: 9)
- Time for RS candidates: 26.25 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 26.25
Output dim: 7, lower bound: -1.5976606, upper bound: 1.6181948
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 10, time: 26.25
Output dim: 7, lower bound: -1.5976606, upper bound: 1.6195556
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 26.25
Output dim: 7, lower bound: -1.5993006, upper bound: 1.6178925
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 10, time: 26.25
Output dim: 7, lower bound: -1.5979006, upper bound: 1.6192554
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 26.25
Output dim: 7, lower bound: -1.6042646, upper bound: 1.5978989
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 26.25
Output dim: 7, lower bound: -1.6042646, upper bound: 1.5993005

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.3005438, 3.2971973
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.5466671, 2.5486116
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.5943165, 2.5949550
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -2.9251938, 2.9313464
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.8534613, 3.8640461
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0254793, 3.0321603
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.0585485, 3.0670362
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0142441, 3.0182300
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.1377759, 2.1285055
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.2690964, 3.2633724

Time for backsubstitution: 13.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5871

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.5868286, upper bound: 1.6195534
time: 5.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.5976563, upper bound: 1.6087166
time: 6.23 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.3021784, 3.2955642
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.5408592, 2.5544202
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.5971680, 2.5921040
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -2.9284630, 2.9280758
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.8534422, 3.8640652
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0263758, 3.0312624
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.0617642, 3.0638227
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0127697, 3.0197039
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.1350741, 2.1312091
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.2702198, 3.2622509

Time for backsubstitution: 13.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5871
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5871

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.5870264, upper bound: 1.6192527
time: 5.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.5978966, upper bound: 1.6084242
time: 5.11 seconds

## Summary of splitting (split count: 10)
- Time for RS candidates: 24.76 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 11, time: 24.76
Output dim: 7, lower bound: -1.5868286, upper bound: 1.6195534
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 11, time: 24.76
Output dim: 7, lower bound: -1.5976563, upper bound: 1.6087166
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 11, time: 24.76
Output dim: 7, lower bound: -1.5870264, upper bound: 1.6192527
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 11, time: 24.76
Output dim: 7, lower bound: -1.5978966, upper bound: 1.6084242

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.2865973, 3.2812552
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.5528183, 2.5557373
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.5889602, 2.5888343
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -2.9224224, 2.9294443
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.8426409, 3.8545780
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0219193, 3.0308790
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.0570464, 3.0653200
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0094204, 3.0140090
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.1171541, 2.1049364
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.2598734, 3.2553015

Time for backsubstitution: 13.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 4572

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.5841724, upper bound: 1.6169513
time: 5.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.5841724, upper bound: 1.6195476
time: 5.25 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.2882299, 3.2796221
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.5470114, 2.5615461
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.5918117, 2.5859833
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -2.9256916, 2.9261737
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.8426208, 3.8545971
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0228148, 3.0299811
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.0602622, 3.0621066
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0079460, 3.0154829
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.1144519, 2.1076400
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.2609959, 3.2541800

Time for backsubstitution: 13.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4572
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 4572

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.5870216, upper bound: 1.6166540
time: 5.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.5843721, upper bound: 1.6192471
time: 5.11 seconds

## Summary of splitting (split count: 11)
- Time for RS candidates: 24.42 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 12, time: 24.42
Output dim: 7, lower bound: -1.5841724, upper bound: 1.6169513
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 12, time: 24.42
Output dim: 7, lower bound: -1.5841724, upper bound: 1.6195476
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 12, time: 24.42
Output dim: 7, lower bound: -1.5870216, upper bound: 1.6166540
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 12, time: 24.42
Output dim: 7, lower bound: -1.5843721, upper bound: 1.6192471

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.2783155, 3.2740078
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.5565624, 2.5600812
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.5908318, 2.5897732
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -2.9290714, 2.9351707
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.8428497, 3.8548107
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0262494, 3.0346093
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.0548024, 3.0627570
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0111904, 3.0155334
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.1167166, 2.1045532
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.2619991, 3.2577729

Time for backsubstitution: 13.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4571
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 523

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.5808488, upper bound: 1.6161841
time: 5.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.5808488, upper bound: 1.6195455
time: 5.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.9743538, -2.1644249, -5.9743538, -2.1644249, -3.2799501, 3.2723742
1: -6.7699232, -3.7522836, -6.7699232, -3.7522836, -2.5507565, 2.5658898
2: -5.1139922, -2.1621265, -5.1139922, -2.1621265, -2.5936832, 2.5869222
3: -8.5304804, -4.5920582, -8.5304804, -4.5920582, -2.9323406, 2.9319000
4: -12.3270998, -8.3508511, -12.3270998, -8.3508511, -3.8428307, 3.8548298
5: -6.8727450, -3.7588763, -6.8727450, -3.7588763, -3.0271459, 3.0337114
6: -10.9138498, -7.3506594, -10.9138498, -7.3506594, -3.0580192, 3.0595431
7: -3.4868472, -0.3735490, -3.4868472, -0.3735490, -3.0097160, 3.0170074
8: 1.5337720, 3.8952756, 1.5337720, 3.8952756, -2.1140144, 2.1072569
9: -8.7088528, -5.0371194, -8.7088528, -5.0371194, -3.2631226, 3.2566509

Time for backsubstitution: 13.84 seconds
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=3.029996871948242
rel_dist={7: [-1.6196747972199932, 1.6196725960818332]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2439.11 seconds
