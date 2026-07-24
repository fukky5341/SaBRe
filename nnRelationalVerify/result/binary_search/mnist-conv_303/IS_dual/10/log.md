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
execution time: IAR + LP analysis = 13.84 + 34.02 = 47.85 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -2.3592026, upper bound: 2.3592013


# Binary Search by BASE starts (time budget: 3552.15 seconds, max iter: 100)

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
Binary search time: 195.94 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Individual Split (IS_dual) starts
Time budget: 3356.21 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 5871
type: B, layer: 1, pos: 5871
type: A, layer: 1, pos: 4571
type: B, layer: 1, pos: 4571
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 5856
type: A, layer: 1, pos: 5856
type: B, layer: 1, pos: 5858
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 4572
type: B, layer: 1, pos: 4572
type: A, layer: 1, pos: 5801
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 4682
type: A, layer: 1, pos: 4682
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 4573
type: A, layer: 1, pos: 4573
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 6139
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 887
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 67

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 565

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9887468, upper bound: 1.9962101
time: 4.88 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9962109, upper bound: 1.9962099
time: 4.63 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 9.71 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 9.71
Output dim: 7, lower bound: -1.9887468, upper bound: 1.9962101
IS_A2, status: Status.UNKNOWN, split count: 1, time: 9.71
Output dim: 7, lower bound: -1.9962109, upper bound: 1.9962099

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -5.9197569, -2.2608802, -5.9702158, -2.1875408, -3.4722872, 3.5240178
1: -6.7016020, -3.7835846, -6.7544537, -3.7534709, -2.9048696, 2.8510740
2: -5.0831227, -2.1913643, -5.1111021, -2.1672659, -2.7666683, 2.7773550
3: -8.4927025, -4.6263618, -8.5273485, -4.6000156, -3.3705111, 3.4011364
4: -12.2562580, -8.3918877, -12.3077936, -8.3543663, -3.9018917, 3.9159060
5: -6.8390732, -3.7869825, -6.8684373, -3.7628384, -3.0762348, 3.0814548
6: -10.8828382, -7.4158897, -10.9108715, -7.3657436, -3.2974315, 3.2286329
7: -3.4275479, -0.4036295, -3.4743609, -0.3756185, -3.0519295, 3.0707314
8: 1.5684128, 3.7872210, 1.5358419, 3.8691750, -2.3007622, 2.2513790
9: -8.6620216, -5.0523887, -8.7030716, -5.0391388, -3.5312424, 3.5460548

Time for backsubstitution: 13.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 5871
type: B, layer: 1, pos: 5871
type: B, layer: 1, pos: 4571
type: A, layer: 1, pos: 4571
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 5856
type: A, layer: 1, pos: 5858
type: B, layer: 1, pos: 5856
type: A, layer: 1, pos: 5801
type: B, layer: 1, pos: 5801
type: A, layer: 1, pos: 4572
type: B, layer: 1, pos: 4572
type: A, layer: 1, pos: 4682
type: B, layer: 1, pos: 5858
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 4682
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 4573
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 887
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 6139
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 67

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 565

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9887468, upper bound: 1.9887457
time: 4.77 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9887468, upper bound: 1.9962100
time: 5.29 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -5.9743538, -2.1644485, -5.9743538, -2.1644249, -3.6227217, 3.5703893
1: -6.7699089, -3.7522840, -6.7699232, -3.7522836, -2.8665218, 2.9016302
2: -5.1139917, -2.1621308, -5.1139922, -2.1621265, -2.8121762, 2.8174775
3: -8.5304775, -4.5920672, -8.5304804, -4.5920582, -3.4166460, 3.3991952
4: -12.3270874, -8.3508549, -12.3270998, -8.3508511, -3.9762363, 3.9762449
5: -6.8727422, -3.7588780, -6.8727450, -3.7588763, -3.1138659, 3.1138670
6: -10.9138508, -7.3506694, -10.9138498, -7.3506594, -3.3483062, 3.3360696
7: -3.4868371, -0.3735507, -3.4868472, -0.3735490, -3.1132882, 3.1132965
8: 1.5337729, 3.8952570, 1.5337720, 3.8952756, -2.3615026, 2.3614850
9: -8.7088499, -5.0371222, -8.7088528, -5.0371194, -3.5754251, 3.5797219

Time for backsubstitution: 13.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5858
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 5871
type: A, layer: 1, pos: 5871
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4571
type: A, layer: 1, pos: 4571
type: A, layer: 1, pos: 5856
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 4682
type: B, layer: 1, pos: 4572
type: A, layer: 1, pos: 4572
type: B, layer: 1, pos: 5801
type: A, layer: 1, pos: 5801
type: B, layer: 1, pos: 5856
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 4682
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 4573
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 887
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 5858

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5858

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9955037, upper bound: 1.9860900
time: 8.13 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9962081, upper bound: 1.9962082
time: 5.29 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 27.34 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 27.34
Output dim: 7, lower bound: -1.9887468, upper bound: 1.9887457
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 27.34
Output dim: 7, lower bound: -1.9887468, upper bound: 1.9962100
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 27.34
Output dim: 7, lower bound: -1.9955037, upper bound: 1.9860900
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 27.34
Output dim: 7, lower bound: -1.9962081, upper bound: 1.9962082

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -5.9197569, -2.2608802, -5.9197569, -2.2608802, -3.4090633, 3.4090629
1: -6.7016020, -3.7835846, -6.7016020, -3.7835846, -2.8691645, 2.8691638
2: -5.0831227, -2.1913643, -5.0831227, -2.1913643, -2.7375965, 2.7375968
3: -8.4927025, -4.6263618, -8.4927025, -4.6263618, -3.3663955, 3.3663964
4: -12.2562580, -8.3918877, -12.2562580, -8.3918877, -3.8643703, 3.8643703
5: -6.8390732, -3.7869825, -6.8390732, -3.7869825, -3.0520906, 3.0520906
6: -10.8828382, -7.4158897, -10.8828382, -7.4158897, -3.1964254, 3.1964247
7: -3.4275479, -0.4036295, -3.4275479, -0.4036295, -3.0239184, 3.0239184
8: 1.5684128, 3.7872210, 1.5684128, 3.7872210, -2.2188082, 2.2188082
9: -8.6620216, -5.0523887, -8.6620216, -5.0523887, -3.5158129, 3.5158119

Time for backsubstitution: 13.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 5871
type: B, layer: 1, pos: 5871
type: A, layer: 1, pos: 4571
type: B, layer: 1, pos: 4571
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 5856
type: B, layer: 1, pos: 5856
type: A, layer: 1, pos: 5801
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 5858
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 4572
type: B, layer: 1, pos: 4572
type: B, layer: 1, pos: 4682
type: A, layer: 1, pos: 4682
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 4573
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 887
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 6139
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 67

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 957

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9868789, upper bound: 1.9639371
time: 4.75 seconds

## Relational analysis of IS_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9868789, upper bound: 1.9868874
time: 4.65 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -5.9197569, -2.2608802, -5.9743538, -2.1644485, -3.4739556, 3.5250356
1: -6.7016020, -3.7835846, -6.7699089, -3.7522840, -2.9068375, 2.8644018
2: -5.0831227, -2.1913643, -5.1139917, -2.1621308, -2.7625284, 2.7798610
3: -8.4927025, -4.6263618, -8.5304775, -4.5920672, -3.3787026, 3.4040670
4: -12.2562580, -8.3918877, -12.3270874, -8.3508549, -3.9054031, 3.9351997
5: -6.8390732, -3.7869825, -6.8727422, -3.7588780, -3.0801952, 3.0857596
6: -10.8828382, -7.4158897, -10.9138508, -7.3506694, -3.3125882, 3.2317820
7: -3.4275479, -0.4036295, -3.4868371, -0.3735507, -3.0539973, 3.0832076
8: 1.5684128, 3.7872210, 1.5337729, 3.8952570, -2.3268442, 2.2534480
9: -8.6620216, -5.0523887, -8.7088499, -5.0371222, -3.5315104, 3.5519910

Time for backsubstitution: 13.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 5871
type: B, layer: 1, pos: 5871
type: A, layer: 1, pos: 5858
type: B, layer: 1, pos: 4571
type: A, layer: 1, pos: 4571
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 5856
type: A, layer: 1, pos: 5801
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 5856
type: A, layer: 1, pos: 4682
type: B, layer: 1, pos: 4572
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 4682
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 4573
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 887
type: B, layer: 1, pos: 5858
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 6139
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 67

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 957

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9639321, upper bound: 1.9943459
time: 4.75 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9868768, upper bound: 1.9943461
time: 5.13 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -5.9228458, -2.1928997, -5.9678411, -2.1713657, -3.5530076, 3.5329585
1: -6.7509212, -3.7711422, -6.7675085, -3.7547774, -2.8349619, 2.8659694
2: -5.0760322, -2.1962886, -5.1076717, -2.1660080, -2.7721434, 2.7759454
3: -8.4774952, -4.6632819, -8.5257816, -4.6104202, -3.3469372, 3.3231754
4: -12.2760792, -8.3918791, -12.3150501, -8.3565750, -3.9195042, 3.9231710
5: -6.8189855, -3.8141303, -6.8670917, -3.7728763, -3.0461092, 3.0529613
6: -10.8785515, -7.4040751, -10.9097023, -7.3636694, -3.2980156, 3.2782381
7: -3.4232366, -0.4064169, -3.4769642, -0.3779624, -3.0452743, 3.0705473
8: 1.5639739, 3.8502154, 1.5375891, 3.8838224, -2.3198485, 2.3126264
9: -8.6601334, -5.0712366, -8.6979637, -5.0407095, -3.5238285, 3.5292010

Time for backsubstitution: 13.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 5871
type: A, layer: 1, pos: 5871
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4571
type: A, layer: 1, pos: 5856
type: A, layer: 1, pos: 4571
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 4682
type: B, layer: 1, pos: 4572
type: A, layer: 1, pos: 4572
type: B, layer: 1, pos: 5801
type: A, layer: 1, pos: 5801
type: B, layer: 1, pos: 5856
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 4682
type: B, layer: 1, pos: 4573
type: A, layer: 1, pos: 4573
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 887
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 565

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 957

## Relational analysis of IS_A2_A1_A1

### Relational analysis result of IS_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9706698, upper bound: 1.9841611
time: 4.97 seconds

## Relational analysis of IS_A2_A1_A2

### Relational analysis result of IS_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9936075, upper bound: 1.9841611
time: 4.98 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -5.9743490, -2.1644549, -5.9743538, -2.1644249, -3.6204567, 3.5703797
1: -6.7699084, -3.7522860, -6.7699232, -3.7522836, -2.8661642, 2.9031649
2: -5.1139884, -2.1621327, -5.1139922, -2.1621265, -2.8106151, 2.8210831
3: -8.5304747, -4.5920739, -8.5304804, -4.5920582, -3.4150515, 3.3550515
4: -12.3270760, -8.3508587, -12.3270998, -8.3508511, -3.9762249, 3.9762411
5: -6.8727369, -3.7588851, -6.8727450, -3.7588763, -3.1138606, 3.1138599
6: -10.9138460, -7.3506823, -10.9138498, -7.3506594, -3.3483043, 3.3302643
7: -3.4868314, -0.3735523, -3.4868472, -0.3735490, -3.1132824, 3.1132948
8: 1.5337758, 3.8952475, 1.5337720, 3.8952756, -2.3614998, 2.3614755
9: -8.7088394, -5.0371246, -8.7088528, -5.0371194, -3.5600767, 3.5797205

Time for backsubstitution: 13.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 5871
type: A, layer: 1, pos: 5871
type: A, layer: 1, pos: 4682
type: A, layer: 1, pos: 5856
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4571
type: A, layer: 1, pos: 4571
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 4572
type: B, layer: 1, pos: 4572
type: B, layer: 1, pos: 5801
type: A, layer: 1, pos: 5801
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 5856
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 4573
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 4573
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 887
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 4682
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 5858

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 957

## Relational analysis of IS_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9943443, upper bound: 1.9713913
time: 7.85 seconds

## Relational analysis of IS_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9943442, upper bound: 1.9943438
time: 5.19 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 26.93 seconds
IS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 26.93
Output dim: 7, lower bound: -1.9868789, upper bound: 1.9639371
IS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 26.93
Output dim: 7, lower bound: -1.9868789, upper bound: 1.9868874
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 26.93
Output dim: 7, lower bound: -1.9639321, upper bound: 1.9943459
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 26.93
Output dim: 7, lower bound: -1.9868768, upper bound: 1.9943461
IS_A2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 26.93
Output dim: 7, lower bound: -1.9706698, upper bound: 1.9841611
IS_A2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 26.93
Output dim: 7, lower bound: -1.9936075, upper bound: 1.9841611
IS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 26.93
Output dim: 7, lower bound: -1.9943443, upper bound: 1.9713913
IS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 26.93
Output dim: 7, lower bound: -1.9943442, upper bound: 1.9943438

## BFS IS instance: IS_A1_B1_B1

### Backsubstitution after applying IS history:
0: -5.9189434, -2.2627707, -5.9056916, -2.2684796, -3.4002161, 3.3927526
1: -6.7010107, -3.7860837, -6.6941595, -3.7956285, -2.8567629, 2.8582847
2: -5.0821214, -2.1946912, -5.0653005, -2.2038851, -2.7241378, 2.7160673
3: -8.4918089, -4.6268468, -8.4888859, -4.6286640, -3.3630805, 3.3630686
4: -12.2507038, -8.3925819, -12.2360144, -8.4126062, -3.8380976, 3.8434324
5: -6.8377495, -3.7871933, -6.8329277, -3.7915423, -3.0462072, 3.0457344
6: -10.8821449, -7.4171515, -10.8735466, -7.4207988, -3.1906252, 3.1851456
7: -3.4195852, -0.4039874, -3.3961625, -0.4290977, -2.9904876, 2.9921751
8: 1.5689430, 3.7821383, 1.5866947, 3.7683005, -2.1993575, 2.1954436
9: -8.6597252, -5.0528541, -8.6516933, -5.0619082, -3.5039749, 3.5053077

Time for backsubstitution: 13.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5871
type: B, layer: 1, pos: 5871
type: A, layer: 1, pos: 4571
type: B, layer: 1, pos: 4571
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 5856
type: A, layer: 1, pos: 5856
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 5801
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 5858
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 4572
type: B, layer: 1, pos: 4572
type: A, layer: 1, pos: 4682
type: B, layer: 1, pos: 4682
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 4573
type: A, layer: 1, pos: 4573
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 887
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 6139
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 67

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5871

## Relational analysis of IS_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5871

## Relational analysis of IS_A1_B1_B1_B1

### Relational analysis result of IS_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9798516, upper bound: 1.9421080
time: 4.90 seconds

## Relational analysis of IS_A1_B1_B1_B2

### Relational analysis result of IS_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9868832, upper bound: 1.9562836
time: 5.27 seconds

## BFS IS instance: IS_A1_B1_B2

### Backsubstitution after applying IS history:
0: -5.9197350, -2.2609153, -5.9196987, -2.2609878, -3.4055119, 3.4089746
1: -6.7015920, -3.7836003, -6.7015843, -3.7836485, -2.8690128, 2.8698573
2: -5.0831223, -2.1914024, -5.0831208, -2.1914828, -2.7293649, 2.7375698
3: -8.4926710, -4.6266565, -8.4926014, -4.6277452, -3.3689570, 3.3660169
4: -12.2562017, -8.3918982, -12.2561073, -8.3919315, -3.8642702, 3.8642092
5: -6.8390651, -3.7869883, -6.8390446, -3.7869945, -3.0520706, 3.0520563
6: -10.8828363, -7.4159212, -10.8828354, -7.4160085, -3.1936598, 3.1963968
7: -3.4275291, -0.4036295, -3.4274657, -0.4036295, -3.0238996, 3.0238361
8: 1.5684190, 3.7872195, 1.5684352, 3.7872143, -2.2187953, 2.2187843
9: -8.6619825, -5.0523987, -8.6618786, -5.0524049, -3.5157480, 3.5107765

Time for backsubstitution: 13.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5871
type: A, layer: 1, pos: 5871
type: B, layer: 1, pos: 4571
type: A, layer: 1, pos: 4571
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 5856
type: B, layer: 1, pos: 5856
type: B, layer: 1, pos: 5801
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 5858
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 4572
type: A, layer: 1, pos: 4572
type: B, layer: 1, pos: 4682
type: A, layer: 1, pos: 4682
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 4573
type: B, layer: 1, pos: 4573
type: A, layer: 1, pos: 887
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6139
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 67

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5871

## Relational analysis of IS_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5871

## Relational analysis of IS_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4571

## Relational analysis of IS_A1_B1_B2_B1

### Relational analysis result of IS_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9727643, upper bound: 1.9833432
time: 4.84 seconds

## Relational analysis of IS_A1_B1_B2_B2

### Relational analysis result of IS_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9868653, upper bound: 1.9868656
time: 5.39 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -5.9056916, -2.2684796, -5.9735079, -2.1663365, -3.4568024, 3.5162990
1: -6.6941595, -3.7956285, -6.7693071, -3.7547779, -2.8959589, 2.8520222
2: -5.0653005, -2.2038851, -5.1130195, -2.1653471, -2.7411175, 2.7664375
3: -8.4888859, -4.6286640, -8.5295773, -4.5925579, -3.3751073, 3.4007506
4: -12.2360144, -8.4126062, -12.3215332, -8.3515587, -3.8844557, 3.9089270
5: -6.8329277, -3.7915423, -6.8714008, -3.7590981, -3.0738297, 3.0798585
6: -10.8735466, -7.4207988, -10.9131098, -7.3519297, -3.3012781, 3.2259285
7: -3.3961625, -0.4290977, -3.4789026, -0.3739064, -3.0222561, 3.0498049
8: 1.5866947, 3.7683005, 1.5342946, 3.8901830, -2.3034883, 2.2340059
9: -8.6516933, -5.0619082, -8.7065363, -5.0376015, -3.5209894, 3.5400152

Time for backsubstitution: 13.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5871
type: B, layer: 1, pos: 5871
type: A, layer: 1, pos: 5858
type: B, layer: 1, pos: 4571
type: A, layer: 1, pos: 4571
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 5856
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 5801
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 5856
type: A, layer: 1, pos: 4682
type: B, layer: 1, pos: 4572
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 4682
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 4573
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 887
type: B, layer: 1, pos: 5858
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 6139
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 67

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5871

## Relational analysis of IS_A1_B2_A1_A1

### Relational analysis result of IS_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9421025, upper bound: 1.9873107
time: 4.67 seconds

## Relational analysis of IS_A1_B2_A1_A2

### Relational analysis result of IS_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9562767, upper bound: 1.9943405
time: 4.90 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -5.9196987, -2.2609878, -5.9743309, -2.1644840, -3.4666824, 3.5183077
1: -6.7015843, -3.7836485, -6.7698965, -3.7522981, -2.9075308, 2.8614101
2: -5.0831208, -2.1914828, -5.1139903, -2.1621690, -2.7624984, 2.7710793
3: -8.4926014, -4.6277452, -8.5304470, -4.5923600, -3.3783283, 3.4066267
4: -12.2561073, -8.3919315, -12.3270321, -8.3508644, -3.9052429, 3.9351006
5: -6.8390446, -3.7869945, -6.8727331, -3.7588849, -3.0801597, 3.0857387
6: -10.8828354, -7.4160085, -10.9138489, -7.3506999, -3.3125596, 3.2290154
7: -3.4274657, -0.4036295, -3.4868169, -0.3735507, -3.0539150, 3.0831873
8: 1.5684352, 3.7872143, 1.5337782, 3.8952556, -2.3268204, 2.2534361
9: -8.6618786, -5.0524049, -8.7088108, -5.0371304, -3.5264769, 3.5519266

Time for backsubstitution: 13.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5871
type: B, layer: 1, pos: 5871
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 4571
type: B, layer: 1, pos: 4571
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 5856
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 5801
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 5856
type: A, layer: 1, pos: 4682
type: A, layer: 1, pos: 4572
type: B, layer: 1, pos: 4572
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 4682
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 4573
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 887
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 6139
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 67

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5871

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5871

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5858

## Relational analysis of IS_A1_B2_A2_A1

### Relational analysis result of IS_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9777445, upper bound: 1.9607828
time: 6.01 seconds

## Relational analysis of IS_A1_B2_A2_A2

### Relational analysis result of IS_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9868753, upper bound: 1.9943440
time: 4.99 seconds

## BFS IS instance: IS_A2_A1_A1

### Backsubstitution after applying IS history:
0: -5.9085994, -2.2005012, -5.9670138, -2.1732540, -3.4733033, 3.5242171
1: -6.7430849, -3.7832291, -6.7669106, -3.7572682, -2.8952732, 2.8535507
2: -5.0581455, -2.2088578, -5.1066928, -2.1692238, -2.7378526, 2.7625256
3: -8.4736681, -4.6657977, -8.5248928, -4.6109114, -3.3368692, 3.3399172
4: -12.2551517, -8.4124298, -12.3094997, -8.3572674, -3.8978844, 3.8970699
5: -6.8128929, -3.8189111, -6.8657699, -3.7730963, -3.0397966, 3.0468588
6: -10.8694935, -7.4092474, -10.9089880, -7.3649311, -3.2868347, 3.2108011
7: -3.3911877, -0.4318435, -3.4690332, -0.3783114, -3.0128763, 3.0371897
8: 1.5822330, 3.8310552, 1.5381103, 3.8787460, -2.2965131, 2.2929449
9: -8.6494722, -5.0805931, -8.6956568, -5.0411758, -3.5171566, 3.5173812

Time for backsubstitution: 13.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5871
type: A, layer: 1, pos: 5871
type: A, layer: 1, pos: 5856
type: B, layer: 1, pos: 4571
type: A, layer: 1, pos: 4571
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 5801
type: B, layer: 1, pos: 5801
type: A, layer: 1, pos: 4682
type: B, layer: 1, pos: 5856
type: B, layer: 1, pos: 4572
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 4573
type: B, layer: 1, pos: 4573
type: A, layer: 1, pos: 887
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 4682
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6139
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 565

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5871

## Relational analysis of IS_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5871

## Relational analysis of IS_A2_A1_A1_A1

### Relational analysis result of IS_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9488531, upper bound: 1.9771969
time: 5.18 seconds

## Relational analysis of IS_A2_A1_A1_A2

### Relational analysis result of IS_A2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9630192, upper bound: 1.9841559
time: 4.77 seconds

## BFS IS instance: IS_A2_A1_A2

### Backsubstitution after applying IS history:
0: -5.9227877, -2.1930077, -5.9678183, -2.1714015, -3.5458884, 3.5273404
1: -6.7509027, -3.7712047, -6.7674975, -3.7547915, -2.8349290, 2.8640485
2: -5.0760293, -2.1964042, -5.1076713, -2.1660452, -2.7703004, 2.7671649
3: -8.4773970, -4.6646671, -8.5257511, -4.6107140, -3.3443990, 3.3230972
4: -12.2759228, -8.3919201, -12.3149967, -8.3565865, -3.9193363, 3.9230766
5: -6.8189583, -3.8141413, -6.8670831, -3.7728848, -3.0460734, 3.0529418
6: -10.8785486, -7.4041924, -10.9097052, -7.3637018, -3.2979865, 3.2768004
7: -3.4231472, -0.4064176, -3.4769433, -0.3779607, -3.0451865, 3.0705256
8: 1.5639973, 3.8502092, 1.5375934, 3.8838210, -2.3198237, 2.3126159
9: -8.6599922, -5.0712514, -8.6979246, -5.0407176, -3.5213671, 3.5291386

Time for backsubstitution: 13.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5871
type: B, layer: 1, pos: 5871
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4571
type: A, layer: 1, pos: 4571
type: A, layer: 1, pos: 5856
type: A, layer: 1, pos: 4682
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 4572
type: A, layer: 1, pos: 4572
type: B, layer: 1, pos: 5801
type: A, layer: 1, pos: 5801
type: B, layer: 1, pos: 5856
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 4682
type: B, layer: 1, pos: 4573
type: A, layer: 1, pos: 4573
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 887
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 565

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5871

## Relational analysis of IS_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5871

## Relational analysis of IS_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4666

## Relational analysis of IS_A2_A1_A2_A1

### Relational analysis result of IS_A2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9911237, upper bound: 1.9826576
time: 5.01 seconds

## Relational analysis of IS_A2_A1_A2_A2

### Relational analysis result of IS_A2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9935892, upper bound: 1.9841424
time: 4.99 seconds

## BFS IS instance: IS_A2_A2_B1

### Backsubstitution after applying IS history:
0: -5.9735065, -2.1663442, -5.9595041, -2.1719961, -3.6117773, 3.5534086
1: -6.7693071, -3.7547796, -6.7618513, -3.7643499, -2.8537893, 2.8910575
2: -5.1130161, -2.1653485, -5.0960102, -2.1744895, -2.7971926, 2.7978785
3: -8.5295753, -4.5925627, -8.5266094, -4.5946884, -3.4095540, 3.3514524
4: -12.3215218, -8.3515606, -12.3058586, -8.3716679, -3.9498539, 3.9542980
5: -6.8713961, -3.7591043, -6.8663845, -3.7638373, -3.1075587, 3.1072803
6: -10.9131079, -7.3519440, -10.9043083, -7.3559141, -3.3421822, 3.3185921
7: -3.4788973, -0.3739066, -3.4550016, -0.3990443, -3.0798531, 3.0810950
8: 1.5342979, 3.8901720, 1.5520363, 3.8760557, -2.3417578, 2.3381357
9: -8.7065277, -5.0376015, -8.6979570, -5.0467744, -3.5479479, 3.5686789

Time for backsubstitution: 13.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5871
type: A, layer: 1, pos: 5871
type: A, layer: 1, pos: 4682
type: A, layer: 1, pos: 5856
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4571
type: A, layer: 1, pos: 4571
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 4572
type: B, layer: 1, pos: 4572
type: B, layer: 1, pos: 5801
type: A, layer: 1, pos: 5801
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 5856
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 4573
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 887
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 4682
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 5858

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5871

## Relational analysis of IS_A2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9873087, upper bound: 1.9495670
time: 4.98 seconds

## Relational analysis of IS_A2_A2_B1_B2

### Relational analysis result of IS_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9943391, upper bound: 1.9637370
time: 5.42 seconds

## BFS IS instance: IS_A2_A2_B2

### Backsubstitution after applying IS history:
0: -5.9743276, -2.1644926, -5.9742966, -2.1645360, -3.6137824, 3.5702920
1: -6.7698965, -3.7523007, -6.7699037, -3.7523470, -2.8642516, 2.8986483
2: -5.1139865, -2.1621718, -5.1139894, -2.1622472, -2.8018341, 2.8108587
3: -8.5304432, -4.5923676, -8.5303783, -4.5934386, -3.4136281, 3.3546753
4: -12.3270206, -8.3508654, -12.3269529, -8.3508968, -3.9761238, 3.9760876
5: -6.8727307, -3.7588918, -6.8727155, -3.7588880, -3.1138427, 3.1138237
6: -10.9138470, -7.3507147, -10.9138489, -7.3507805, -3.3468647, 3.3302355
7: -3.4868124, -0.3735526, -3.4867585, -0.3735511, -3.1132612, 3.1132059
8: 1.5337806, 3.8952460, 1.5337944, 3.8952680, -2.3614874, 2.3614516
9: -8.7087975, -5.0371323, -8.7087097, -5.0371342, -3.5600128, 3.5772543

Time for backsubstitution: 13.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5871
type: A, layer: 1, pos: 4682
type: A, layer: 1, pos: 5856
type: A, layer: 1, pos: 5871
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4571
type: A, layer: 1, pos: 4571
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 4572
type: A, layer: 1, pos: 4572
type: B, layer: 1, pos: 5801
type: A, layer: 1, pos: 5801
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 5856
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 4573
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 4573
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 887
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 4682
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 5858

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5871

## Relational analysis of IS_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4682

## Relational analysis of IS_A2_A2_B2_A1

### Relational analysis result of IS_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9939101, upper bound: 1.9943426
time: 5.90 seconds

## Relational analysis of IS_A2_A2_B2_A2

### Relational analysis result of IS_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9943428, upper bound: 1.9943424
time: 5.01 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 30.64 seconds
IS_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 30.64
Output dim: 7, lower bound: -1.9798516, upper bound: 1.9421080
IS_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 30.64
Output dim: 7, lower bound: -1.9868832, upper bound: 1.9562836
IS_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 30.64
Output dim: 7, lower bound: -1.9727643, upper bound: 1.9833432
IS_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 30.64
Output dim: 7, lower bound: -1.9868653, upper bound: 1.9868656
IS_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 30.64
Output dim: 7, lower bound: -1.9421025, upper bound: 1.9873107
IS_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 30.64
Output dim: 7, lower bound: -1.9562767, upper bound: 1.9943405
IS_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 30.64
Output dim: 7, lower bound: -1.9777445, upper bound: 1.9607828
IS_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 30.64
Output dim: 7, lower bound: -1.9868753, upper bound: 1.9943440
IS_A2_A1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 30.64
Output dim: 7, lower bound: -1.9488531, upper bound: 1.9771969
IS_A2_A1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 30.64
Output dim: 7, lower bound: -1.9630192, upper bound: 1.9841559
IS_A2_A1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 30.64
Output dim: 7, lower bound: -1.9911237, upper bound: 1.9826576
IS_A2_A1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 30.64
Output dim: 7, lower bound: -1.9935892, upper bound: 1.9841424
IS_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 30.64
Output dim: 7, lower bound: -1.9873087, upper bound: 1.9495670
IS_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 30.64
Output dim: 7, lower bound: -1.9943391, upper bound: 1.9637370
IS_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 30.64
Output dim: 7, lower bound: -1.9939101, upper bound: 1.9943426
IS_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 30.64
Output dim: 7, lower bound: -1.9943428, upper bound: 1.9943424

## BFS IS instance: IS_A1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -5.9165835, -2.2679186, -5.8874602, -2.2896333, -3.3754554, 3.3685031
1: -6.6992769, -3.7911093, -6.6835418, -3.8180575, -2.8330145, 2.8346252
2: -5.0799174, -2.2050309, -5.0456958, -2.2454433, -2.6789255, 2.6854455
3: -8.4871702, -4.6277966, -8.4665871, -4.6347704, -3.3521600, 3.3398046
4: -12.2359695, -8.3944864, -12.1776543, -8.4370928, -3.7988768, 3.7831678
5: -6.8334160, -3.7880499, -6.8141794, -3.7987912, -3.0346248, 3.0261295
6: -10.8805199, -7.4198852, -10.8624945, -7.4322367, -3.1758990, 3.1681707
7: -3.4032710, -0.4049797, -3.3302121, -0.4507627, -2.9525082, 2.9252324
8: 1.5705686, 3.7726765, 1.6046009, 3.7312689, -2.1607003, 2.1680756
9: -8.6519938, -5.0538740, -8.6183987, -5.0744371, -3.4835615, 3.4719186

Time for backsubstitution: 13.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4571
type: A, layer: 1, pos: 4571
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 5871
type: A, layer: 1, pos: 5856
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 5856
type: A, layer: 1, pos: 5801
type: B, layer: 1, pos: 5801
type: A, layer: 1, pos: 5858
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 4572
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 4682
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 4682
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 4573
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 6139
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 67

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 4571

## Relational analysis of IS_A1_B1_B1_B1_B1

### Relational analysis result of IS_A1_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9657071, upper bound: 1.9385636
time: 5.43 seconds

## Relational analysis of IS_A1_B1_B1_B1_B2

### Relational analysis result of IS_A1_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9798285, upper bound: 1.9420857
time: 5.01 seconds

## BFS IS instance: IS_A1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -5.9189434, -2.2627707, -5.9056911, -2.2684817, -3.3940687, 3.3927517
1: -6.7010107, -3.7860837, -6.6941576, -3.7956314, -2.8567576, 2.8759441
2: -5.0821214, -2.1946912, -5.0653000, -2.2038903, -2.7211595, 2.7160668
3: -8.4918089, -4.6268468, -8.4888821, -4.6286669, -3.3630800, 3.3617105
4: -12.2507038, -8.3925819, -12.2360115, -8.4126053, -3.8380985, 3.8434296
5: -6.8377495, -3.7871933, -6.8329258, -3.7915416, -3.0462079, 3.0457325
6: -10.8821449, -7.4171515, -10.8735456, -7.4207988, -3.1861086, 3.1851451
7: -3.4195852, -0.4039874, -3.3961575, -0.4290972, -2.9904881, 2.9921701
8: 1.5689430, 3.7821383, 1.5866957, 3.7682953, -2.1993523, 2.1954427
9: -8.6597252, -5.0528541, -8.6516924, -5.0619087, -3.5039759, 3.4923782

Time for backsubstitution: 13.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4571
type: A, layer: 1, pos: 4571
type: A, layer: 1, pos: 5871
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 5856
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 5801
type: A, layer: 1, pos: 5801
type: B, layer: 1, pos: 5856
type: A, layer: 1, pos: 5858
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 4572
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 4682
type: B, layer: 1, pos: 4682
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 4573
type: B, layer: 1, pos: 4573
type: A, layer: 1, pos: 887
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 6139
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 67

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 4571

## Relational analysis of IS_A1_B1_B1_B2_B1

### Relational analysis result of IS_A1_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9727591, upper bound: 1.9527389
time: 4.69 seconds

## Relational analysis of IS_A1_B1_B1_B2_B2

### Relational analysis result of IS_A1_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9868602, upper bound: 1.9562614
time: 5.14 seconds

## BFS IS instance: IS_A1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -5.9155617, -2.2648890, -5.9027705, -2.2776971, -3.3855333, 3.3790083
1: -6.6888046, -3.7860219, -6.6535830, -3.8124890, -2.8276091, 2.7739873
2: -5.0802445, -2.1976094, -5.0642934, -2.2158737, -2.7119236, 2.7097569
3: -8.4887905, -4.6393442, -8.4518223, -4.6726613, -3.3212805, 3.3334212
4: -12.2460384, -8.3963118, -12.2196074, -8.4204979, -3.8255405, 3.8232956
5: -6.8367939, -3.7931881, -6.8257408, -3.8106573, -3.0117798, 3.0325527
6: -10.8649807, -7.4188108, -10.8180847, -7.4593349, -3.1310759, 3.1121154
7: -3.4219198, -0.4118009, -3.3923311, -0.4331405, -2.9887793, 2.9805303
8: 1.5719881, 3.7783337, 1.5996127, 3.7556610, -2.1836729, 2.1787210
9: -8.6573563, -5.0554094, -8.6349277, -5.0636787, -3.4945230, 3.4414697

Time for backsubstitution: 13.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5871
type: A, layer: 1, pos: 5871
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 5856
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 5858
type: A, layer: 1, pos: 5801
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 5856
type: B, layer: 1, pos: 4682
type: A, layer: 1, pos: 4571
type: B, layer: 1, pos: 4572
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 4682
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 887
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 67

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5871

## Relational analysis of IS_A1_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5871

## Relational analysis of IS_A1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4666

## Relational analysis of IS_A1_B1_B2_B1_A1

### Relational analysis result of IS_A1_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9702603, upper bound: 1.9828094
time: 4.41 seconds

## Relational analysis of IS_A1_B1_B2_B1_A2

### Relational analysis result of IS_A1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9727509, upper bound: 1.9833290
time: 4.93 seconds

## BFS IS instance: IS_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -5.9197311, -2.2609191, -5.9196835, -2.2610102, -3.4054775, 3.4090352
1: -6.7015781, -3.7836022, -6.7015123, -3.7836576, -2.8689919, 2.8484333
2: -5.0831203, -2.1914077, -5.0831065, -2.1915102, -2.7276797, 2.7375467
3: -8.4926691, -4.6266699, -8.4925861, -4.6278214, -3.3365107, 3.3659859
4: -12.2561932, -8.3919020, -12.2560539, -8.3919487, -3.8642445, 3.8641520
5: -6.8390632, -3.7869949, -6.8390374, -3.7870221, -3.0520411, 3.0520425
6: -10.8828163, -7.4159231, -10.8827438, -7.4160185, -3.1936321, 3.1488152
7: -3.4275258, -0.4036362, -3.4274440, -0.4036636, -3.0238621, 3.0238078
8: 1.5684218, 3.7872124, 1.5684471, 3.7871747, -2.2187529, 2.2187653
9: -8.6619778, -5.0523992, -8.6618586, -5.0524139, -3.5167389, 3.5107498

Time for backsubstitution: 13.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5871
type: A, layer: 1, pos: 5871
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 5856
type: B, layer: 1, pos: 5856
type: B, layer: 1, pos: 5801
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 4571
type: A, layer: 1, pos: 5858
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 4572
type: A, layer: 1, pos: 4572
type: B, layer: 1, pos: 4682
type: A, layer: 1, pos: 4682
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 4573
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 887
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6139
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 67

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5871

## Relational analysis of IS_A1_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5871

## Relational analysis of IS_A1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4666

## Relational analysis of IS_A1_B1_B2_B2_B1

### Relational analysis result of IS_A1_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9861257, upper bound: 1.9843589
time: 5.02 seconds

## Relational analysis of IS_A1_B1_B2_B2_B2

### Relational analysis result of IS_A1_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9868507, upper bound: 1.9868507
time: 5.04 seconds

## BFS IS instance: IS_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -5.8874602, -2.2896333, -5.9709764, -2.1714878, -3.4271936, 3.4915259
1: -6.6835418, -3.8180575, -6.7675405, -3.7598019, -2.8722696, 2.8277831
2: -5.0456958, -2.2454433, -5.1107860, -2.1751766, -2.7110472, 2.7212286
3: -8.4665871, -4.6347704, -8.5249519, -4.5935216, -3.3517728, 3.3898873
4: -12.1776543, -8.4370928, -12.3068218, -8.3535395, -3.8241148, 3.8697290
5: -6.8141794, -3.7987912, -6.8673382, -3.7599745, -3.0542049, 3.0685470
6: -10.8624945, -7.4322367, -10.9113140, -7.3546624, -3.2843838, 3.2109978
7: -3.3302121, -0.4507627, -3.4627073, -0.3749287, -2.9552834, 3.0119445
8: 1.6046009, 3.7312689, 1.5359421, 3.8807354, -2.2761345, 2.1953268
9: -8.6183987, -5.0744371, -8.6987400, -5.0387278, -3.4874878, 3.5192342

Time for backsubstitution: 13.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5858
type: B, layer: 1, pos: 4571
type: A, layer: 1, pos: 4571
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 5856
type: B, layer: 1, pos: 5871
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 5801
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 5856
type: A, layer: 1, pos: 4572
type: B, layer: 1, pos: 4572
type: A, layer: 1, pos: 4682
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 4682
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 4573
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 5858
type: A, layer: 1, pos: 887
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 6139
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 67

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5858

## Relational analysis of IS_A1_B2_A1_A1_A1

### Relational analysis result of IS_A1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9329472, upper bound: 1.9543132
time: 5.03 seconds

## Relational analysis of IS_A1_B2_A1_A1_A2

### Relational analysis result of IS_A1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9420967, upper bound: 1.9873085
time: 5.21 seconds

## BFS IS instance: IS_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -5.9056911, -2.2684817, -5.9735079, -2.1663365, -3.4509544, 3.5050011
1: -6.6941576, -3.7956314, -6.7693071, -3.7547779, -2.9136181, 2.8520207
2: -5.0653000, -2.2038903, -5.1130195, -2.1653471, -2.7411165, 2.7620964
3: -8.4888821, -4.6286669, -8.5295773, -4.5925579, -3.3701601, 3.4007502
4: -12.2360115, -8.4126053, -12.3215332, -8.3515587, -3.8844528, 3.9089279
5: -6.8329258, -3.7915416, -6.8714008, -3.7590981, -3.0738277, 3.0798593
6: -10.8735456, -7.4207988, -10.9131098, -7.3519297, -3.3012781, 3.2214110
7: -3.3961575, -0.4290972, -3.4789026, -0.3739064, -3.0222511, 3.0498054
8: 1.5866957, 3.7682953, 1.5342946, 3.8901830, -2.3034873, 2.2340007
9: -8.6516924, -5.0619087, -8.7065363, -5.0376015, -3.5080605, 3.5400147

Time for backsubstitution: 13.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 4571
type: B, layer: 1, pos: 4571
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 5871
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 5856
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 5856
type: A, layer: 1, pos: 5801
type: B, layer: 1, pos: 5801
type: A, layer: 1, pos: 4682
type: A, layer: 1, pos: 4572
type: B, layer: 1, pos: 4572
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 4682
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 4573
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 887
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 6139
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 67

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5858

## Relational analysis of IS_A1_B2_A1_A2_A1

### Relational analysis result of IS_A1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9471379, upper bound: 1.9607781
time: 6.22 seconds

## Relational analysis of IS_A1_B2_A1_A2_A2

### Relational analysis result of IS_A1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9562715, upper bound: 1.9943393
time: 5.48 seconds

## BFS IS instance: IS_A1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -5.8429074, -2.2953506, -5.9670024, -2.1715479, -3.3576851, 3.4715157
1: -6.6794844, -3.8274314, -6.7665687, -3.7550747, -2.8065057, 2.7842429
2: -5.0096784, -2.2296596, -5.1064873, -2.1677122, -2.6837335, 2.7022514
3: -8.4317894, -4.7013106, -8.5251074, -4.6107645, -3.2930346, 3.3282204
4: -12.2109814, -8.5079851, -12.3022194, -8.3574209, -3.8535604, 3.7942343
5: -6.7341471, -3.8409767, -6.8658056, -3.7789803, -2.9551668, 3.0248289
6: -10.7958527, -7.4631920, -10.9094477, -7.3749399, -3.1981497, 3.1672943
7: -3.3583684, -0.5092964, -3.4639802, -0.3783145, -2.9800539, 2.9546838
8: 1.6148210, 3.7378044, 1.5381842, 3.8837538, -2.2689328, 2.1996202
9: -8.6106596, -5.1500382, -8.6889820, -5.0409713, -3.4596386, 3.4292898

Time for backsubstitution: 13.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5871
type: A, layer: 1, pos: 5871
type: B, layer: 1, pos: 4571
type: A, layer: 1, pos: 4571
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 5856
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 5801
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 4682
type: B, layer: 1, pos: 5856
type: A, layer: 1, pos: 4572
type: B, layer: 1, pos: 4572
type: B, layer: 1, pos: 5858
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 4682
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 4573
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 6139
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 67

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5871

## Relational analysis of IS_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5871

## Relational analysis of IS_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4571

## Relational analysis of IS_A1_B2_A2_A1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9635412, upper bound: 1.9569873
time: 5.95 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2

### Relational analysis result of IS_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9777216, upper bound: 1.9607603
time: 4.68 seconds

## BFS IS instance: IS_A1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -5.9196944, -2.2609978, -5.9743309, -2.1644840, -3.4667482, 3.5183058
1: -6.7015829, -3.7836499, -6.7698965, -3.7522981, -2.9028482, 2.8548119
2: -5.0831161, -2.1914868, -5.1139903, -2.1621690, -2.7614932, 2.7695441
3: -8.4925995, -4.6277528, -8.5304470, -4.5923600, -3.3683376, 3.3603387
4: -12.2561035, -8.3919353, -12.3270321, -8.3508644, -3.9052391, 3.9350967
5: -6.8390408, -3.7870026, -6.8727331, -3.7588849, -3.0801558, 3.0857306
6: -10.8828335, -7.4160218, -10.9138489, -7.3506999, -3.3081088, 3.2212539
7: -3.4274609, -0.4036314, -3.4868169, -0.3735507, -3.0539103, 3.0831854
8: 1.5684366, 3.7872014, 1.5337782, 3.8952556, -2.3268189, 2.2534232
9: -8.6618671, -5.0524068, -8.7088108, -5.0371304, -3.5267925, 3.5519257

Time for backsubstitution: 13.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5871
type: B, layer: 1, pos: 5871
type: B, layer: 1, pos: 4571
type: A, layer: 1, pos: 4571
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 5856
type: A, layer: 1, pos: 4682
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 5801
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 5856
type: A, layer: 1, pos: 4572
type: B, layer: 1, pos: 4572
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 4573
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 6139
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 4682
type: B, layer: 1, pos: 5858

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5871

## Relational analysis of IS_A1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5871

## Relational analysis of IS_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4571

## Relational analysis of IS_A1_B2_A2_A2_B1

### Relational analysis result of IS_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9727510, upper bound: 1.9908009
time: 5.25 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2

### Relational analysis result of IS_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9868523, upper bound: 1.9943221
time: 4.96 seconds

## BFS IS instance: IS_A2_A1_A1_A1

### Backsubstitution after applying IS history:
0: -5.8913717, -2.2216468, -5.9645805, -2.1784024, -3.4449263, 3.4994092
1: -6.7323446, -3.8057275, -6.7651482, -3.7622874, -2.8714962, 2.8293343
2: -5.0389628, -2.2489591, -5.1044912, -2.1790533, -2.7038403, 2.7172823
3: -8.4512444, -4.6719294, -8.5201960, -4.6118741, -3.3138638, 3.3288450
4: -12.1966591, -8.4359579, -12.2947693, -8.3591728, -3.8374863, 3.8588114
5: -6.7942362, -3.8262503, -6.8614397, -3.7739742, -3.0202620, 3.0351894
6: -10.8595181, -7.4207335, -10.9072819, -7.3676682, -3.2708621, 3.1959791
7: -3.3253951, -0.4530821, -3.4528487, -0.3792973, -2.9460979, 2.9997666
8: 1.5999193, 3.7940550, 1.5397320, 3.8692961, -2.2693768, 2.2543230
9: -8.6159973, -5.0922866, -8.6878643, -5.0422325, -3.4832287, 3.4974184

Time for backsubstitution: 13.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4571
type: A, layer: 1, pos: 4571
type: A, layer: 1, pos: 5856
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 5871
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 5801
type: B, layer: 1, pos: 5801
type: A, layer: 1, pos: 4682
type: B, layer: 1, pos: 5856
type: B, layer: 1, pos: 4572
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 4573
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 4682
type: A, layer: 1, pos: 887
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 6139
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 565

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 4571

## Relational analysis of IS_A2_A1_A1_A1_B1

### Relational analysis result of IS_A2_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9346535, upper bound: 1.9736171
time: 4.88 seconds

## Relational analysis of IS_A2_A1_A1_A1_B2

### Relational analysis result of IS_A2_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9488304, upper bound: 1.9771748
time: 5.09 seconds

## BFS IS instance: IS_A2_A1_A1_A2

### Backsubstitution after applying IS history:
0: -5.9085979, -2.2005019, -5.9670138, -2.1732540, -3.4677052, 3.5129275
1: -6.7430840, -3.7832313, -6.7669106, -3.7572682, -2.9129267, 2.8535461
2: -5.0581431, -2.2088618, -5.1066928, -2.1692238, -2.7357702, 2.7581844
3: -8.4736662, -4.6658001, -8.5248928, -4.6109114, -3.3319292, 3.3399167
4: -12.2551489, -8.4124298, -12.3094997, -8.3572674, -3.8978815, 3.8970699
5: -6.8128910, -3.8189113, -6.8657699, -3.7730963, -3.0397947, 3.0468585
6: -10.8694925, -7.4092474, -10.9089880, -7.3649311, -3.2868338, 3.2062819
7: -3.3911839, -0.4318430, -3.4690332, -0.3783114, -3.0128725, 3.0371902
8: 1.5822339, 3.8310499, 1.5381103, 3.8787460, -2.2965121, 2.2864997
9: -8.6494694, -5.0805926, -8.6956568, -5.0411758, -3.5042267, 3.5173817

Time for backsubstitution: 13.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5871
type: B, layer: 1, pos: 4571
type: A, layer: 1, pos: 4571
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 5856
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 5801
type: B, layer: 1, pos: 5801
type: A, layer: 1, pos: 4682
type: B, layer: 1, pos: 5856
type: B, layer: 1, pos: 4572
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 4573
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 887
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 4682
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 6139
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 565

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5871

## Relational analysis of IS_A2_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4571

## Relational analysis of IS_A2_A1_A1_A2_B1

### Relational analysis result of IS_A2_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9488227, upper bound: 1.9806088
time: 5.29 seconds

## Relational analysis of IS_A2_A1_A1_A2_B2

### Relational analysis result of IS_A2_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9629965, upper bound: 1.9841340
time: 5.48 seconds

## BFS IS instance: IS_A2_A1_A2_A1

### Backsubstitution after applying IS history:
0: -5.9136395, -2.2052503, -5.9678183, -2.1714015, -3.5362792, 3.5094857
1: -6.7477169, -3.7881374, -6.7674975, -3.7547915, -2.8246589, 2.8445251
2: -5.0649061, -2.2086620, -5.1076713, -2.1660452, -2.7567906, 2.7454674
3: -8.4663467, -4.6682954, -8.5257511, -4.6107140, -3.3303018, 3.3123589
4: -12.2423115, -8.3969555, -12.3149967, -8.3565865, -3.8857250, 3.9180412
5: -6.8087111, -3.8169808, -6.8670831, -3.7728848, -3.0358262, 3.0501022
6: -10.8709583, -7.4064465, -10.9097052, -7.3637018, -3.2872214, 3.2696826
7: -3.4140296, -0.4123802, -3.4769433, -0.3779607, -3.0360689, 3.0645630
8: 1.5696321, 3.8415956, 1.5375934, 3.8838210, -2.3141890, 2.3040023
9: -8.6221581, -5.0766716, -8.6979246, -5.0407176, -3.4873552, 3.5246582

Time for backsubstitution: 13.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5871
type: B, layer: 1, pos: 5871
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 4571
type: A, layer: 1, pos: 4571
type: A, layer: 1, pos: 4682
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 5856
type: A, layer: 1, pos: 5856
type: A, layer: 1, pos: 4572
type: B, layer: 1, pos: 4572
type: B, layer: 1, pos: 5801
type: A, layer: 1, pos: 5801
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 4682
type: A, layer: 1, pos: 4573
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 887
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 565

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5871

## Relational analysis of IS_A2_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5871

## Relational analysis of IS_A2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4666

## Relational analysis of IS_A2_A1_A2_A1_B1

### Relational analysis result of IS_A2_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9901306, upper bound: 1.9803654
time: 5.15 seconds

## Relational analysis of IS_A2_A1_A2_A1_B2

### Relational analysis result of IS_A2_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.9901304, upper bound: 1.9803654
time: 4.81 seconds

## BFS IS instance: IS_A2_A1_A2_A2

### Backsubstitution after applying IS history:
0: -5.9281759, -2.1811576, -5.9678020, -2.1714177, -3.5514560, 3.5345130
1: -6.7728209, -3.7632604, -6.7674904, -3.7548203, -2.8496222, 2.8791173
2: -5.0793686, -2.1728745, -5.1076488, -2.1660590, -2.7754164, 2.7840025
3: -8.4842968, -4.6616373, -8.5257349, -4.6107211, -3.3537235, 3.3188634
4: -12.2940168, -8.3576269, -12.3149385, -8.3565979, -3.9374189, 3.9573116
5: -6.8273535, -3.8109646, -6.8670673, -3.7728887, -3.0544648, 3.0561028
6: -10.8843975, -7.3964601, -10.9096889, -7.3637042, -3.3119578, 3.2802167
7: -3.4385092, -0.4005115, -3.4769285, -0.3779728, -3.0605364, 3.0764170
8: 1.5410848, 3.8534970, 1.5376039, 3.8838015, -2.3427167, 2.3158932
9: -8.6732626, -5.0417891, -8.6978617, -5.0407300, -3.5354252, 3.5590510

Time for backsubstitution: 13.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5871
type: B, layer: 1, pos: 5871
type: A, layer: 1, pos: 4571
type: B, layer: 1, pos: 4571
type: A, layer: 1, pos: 5856
type: A, layer: 1, pos: 4682
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 4572
type: B, layer: 1, pos: 4572
type: B, layer: 1, pos: 5856
type: B, layer: 1, pos: 5801
type: A, layer: 1, pos: 5801
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 4682
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 6139
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 887
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 5858
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 565

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5871

## Relational analysis of IS_A2_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=3.113298177719116
rel_dist={7: [-1.996222388869347, 1.996221151991417]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 5871
type: A, layer: 1, pos: 5871
type: A, layer: 1, pos: 4571
type: B, layer: 1, pos: 4571
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4572
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 5801
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 5856
type: A, layer: 1, pos: 5856
type: B, layer: 1, pos: 4682
type: A, layer: 1, pos: 4682
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 5858
type: A, layer: 1, pos: 5858
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 4573
type: B, layer: 1, pos: 4573
type: A, layer: 1, pos: 887
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 67

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 565

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7159505, upper bound: 1.7205214
time: 5.22 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7205220, upper bound: 1.7205215
time: 4.88 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 10.29 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 10.29
Output dim: 7, lower bound: -1.7159505, upper bound: 1.7205214
IS_A2, status: Status.UNKNOWN, split count: 1, time: 10.29
Output dim: 7, lower bound: -1.7205220, upper bound: 1.7205215

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -5.9197569, -2.2608802, -5.9673557, -2.2032933, -3.2853913, 3.3615298
1: -6.7016020, -3.7835846, -6.7439470, -3.7542973, -2.7335215, 2.6812389
2: -5.0831227, -2.1913643, -5.1090937, -2.1705210, -2.6376147, 2.6427698
3: -8.4927025, -4.6263618, -8.5252237, -4.6054382, -3.1026068, 3.1256647
4: -12.2562580, -8.3918877, -12.2971468, -8.3567448, -3.8995132, 3.9052591
5: -6.8390732, -3.7869825, -6.8655038, -3.7658577, -3.0539594, 3.0476985
6: -10.8828382, -7.4158897, -10.9088039, -7.3760304, -3.0786190, 3.0201621
7: -3.4275479, -0.4036295, -3.4663680, -0.3770251, -3.0394220, 3.0138040
8: 1.5684128, 3.7872210, 1.5372372, 3.8514118, -2.2551219, 2.1974404
9: -8.6620216, -5.0523887, -8.6987162, -5.0405402, -3.3296709, 3.3529615

Time for backsubstitution: 13.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 5871
type: B, layer: 1, pos: 5871
type: A, layer: 1, pos: 4571
type: B, layer: 1, pos: 4571
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 5801
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 4572
type: A, layer: 1, pos: 4572
type: B, layer: 1, pos: 5856
type: A, layer: 1, pos: 5856
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 5858
type: A, layer: 1, pos: 5858
type: B, layer: 1, pos: 4682
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 4682
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 4573
type: A, layer: 1, pos: 4573
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 887
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 6139
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 67

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 957

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7016878, upper bound: 1.7186307
time: 4.83 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7140434, upper bound: 1.7186293
time: 4.95 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -5.9743538, -2.1644485, -5.9743547, -2.1644275, -3.4583235, 3.4006939
1: -6.7699089, -3.7522840, -6.7699223, -3.7522831, -2.6997147, 2.7420642
2: -5.1139917, -2.1621308, -5.1139936, -2.1621265, -2.6786928, 2.6785607
3: -8.5304775, -4.5920672, -8.5304775, -4.5920596, -3.1543221, 3.1337986
4: -12.3270874, -8.3508549, -12.3270979, -8.3508511, -3.9658222, 3.9762430
5: -6.8727422, -3.7588780, -6.8727446, -3.7588763, -3.1138659, 3.0954599
6: -10.9138508, -7.3506694, -10.9138508, -7.3506613, -3.1398377, 3.1254449
7: -3.4868371, -0.3735507, -3.4868457, -0.3735499, -3.0552025, 3.0602515
8: 1.5337729, 3.8952570, 1.5337710, 3.8952742, -2.3480284, 2.2674434
9: -8.7088499, -5.0371222, -8.7088518, -5.0371189, -3.3863211, 3.3861880

Time for backsubstitution: 13.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 5871
type: A, layer: 1, pos: 5871
type: B, layer: 1, pos: 4571
type: A, layer: 1, pos: 4571
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 5856
type: A, layer: 1, pos: 4682
type: B, layer: 1, pos: 4572
type: A, layer: 1, pos: 4572
type: B, layer: 1, pos: 5801
type: A, layer: 1, pos: 5801
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 5856
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 6139
type: B, layer: 1, pos: 4682
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 887
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 565

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 957

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7062767, upper bound: 1.7186306
time: 5.33 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7186292, upper bound: 1.7186306
time: 5.09 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 24.38 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 24.38
Output dim: 7, lower bound: -1.7016878, upper bound: 1.7186307
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 24.38
Output dim: 7, lower bound: -1.7140434, upper bound: 1.7186293
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 24.38
Output dim: 7, lower bound: -1.7062767, upper bound: 1.7186306
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 24.38
Output dim: 7, lower bound: -1.7186292, upper bound: 1.7186306

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -5.9056916, -2.2684796, -5.9658957, -2.2066092, -3.2667999, 3.3521924
1: -6.6941595, -3.7956285, -6.7428970, -3.7586703, -2.7206354, 2.6683753
2: -5.0653005, -2.2038851, -5.1073904, -2.1762285, -2.6135278, 2.6287193
3: -8.4888859, -4.6286640, -8.5236673, -4.6062951, -3.0986762, 3.1216197
4: -12.2360144, -8.4126062, -12.2873974, -8.3579769, -3.8780375, 3.8747911
5: -6.8329277, -3.7915423, -6.8631630, -3.7662365, -3.0410275, 3.0377440
6: -10.8735466, -7.4207988, -10.9075375, -7.3782392, -3.0663643, 3.0136814
7: -3.3961625, -0.4290977, -3.4524369, -0.3776522, -3.0076160, 2.9735935
8: 1.5866947, 3.7683005, 1.5381618, 3.8424907, -2.2241886, 2.1772778
9: -8.6516933, -5.0619082, -8.6946659, -5.0413728, -3.3187819, 3.3392472

Time for backsubstitution: 13.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5871
type: B, layer: 1, pos: 5871
type: A, layer: 1, pos: 4571
type: B, layer: 1, pos: 4571
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 5801
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 4572
type: A, layer: 1, pos: 4572
type: B, layer: 1, pos: 5856
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 5856
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 5858
type: A, layer: 1, pos: 5858
type: B, layer: 1, pos: 4682
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 4682
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 4573
type: A, layer: 1, pos: 4573
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 887
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 6139
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 67

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5871

## Relational analysis of IS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5871

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4571

## Relational analysis of IS_A1_A1_A1

### Relational analysis result of IS_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6962057, upper bound: 1.7069620
time: 5.38 seconds

## Relational analysis of IS_A1_A1_A2

### Relational analysis result of IS_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7016701, upper bound: 1.7186141
time: 5.45 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -5.9196987, -2.2609878, -5.9673152, -2.2033725, -3.2780771, 3.3549242
1: -6.7015843, -3.7836485, -6.7439299, -3.7543287, -2.7340879, 2.6789956
2: -5.0831208, -2.1914828, -5.1090899, -2.1706033, -2.6375566, 2.6324513
3: -8.4926014, -4.6277452, -8.5251646, -4.6060920, -3.1018705, 3.1278129
4: -12.2561073, -8.3919315, -12.2970324, -8.3567657, -3.8993416, 3.9051008
5: -6.8390446, -3.7869945, -6.8654833, -3.7658699, -3.0487814, 3.0420690
6: -10.8828354, -7.4160085, -10.9088030, -7.3760953, -3.0785666, 3.0169215
7: -3.4274657, -0.4036295, -3.4663255, -0.3770263, -3.0299921, 2.9988654
8: 1.5684352, 3.7872143, 1.5372496, 3.8514085, -2.2409806, 2.1785929
9: -8.6618786, -5.0524049, -8.6986227, -5.0405507, -3.3237743, 3.3528347

Time for backsubstitution: 13.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5871
type: B, layer: 1, pos: 5871
type: A, layer: 1, pos: 4571
type: B, layer: 1, pos: 4571
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 5801
type: B, layer: 1, pos: 5801
type: A, layer: 1, pos: 4572
type: B, layer: 1, pos: 4572
type: B, layer: 1, pos: 5856
type: A, layer: 1, pos: 5856
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 4682
type: A, layer: 1, pos: 5858
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 4682
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 4573
type: A, layer: 1, pos: 4573
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 6139
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 67

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5871

## Relational analysis of IS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5871

## Relational analysis of IS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4571

## Relational analysis of IS_A1_A2_A1

### Relational analysis result of IS_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7085495, upper bound: 1.7069619
time: 5.10 seconds

## Relational analysis of IS_A1_A2_A2

### Relational analysis result of IS_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7140274, upper bound: 1.7186131
time: 5.26 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -5.9595032, -2.1720204, -5.9728684, -2.1677439, -3.4391265, 3.3913836
1: -6.7618365, -3.7643509, -6.7688646, -3.7566581, -2.6866760, 2.7292087
2: -5.0960088, -2.1744928, -5.1123128, -2.1677728, -2.6544394, 2.6644754
3: -8.5266075, -4.5946965, -8.5288954, -4.5929184, -3.1503859, 3.1293554
4: -12.3058434, -8.3716679, -12.3173475, -8.3520889, -3.9435329, 3.9456797
5: -6.8663836, -3.7638397, -6.8703852, -3.7592578, -3.1071258, 3.0844746
6: -10.9043083, -7.3559241, -10.9125490, -7.3528681, -3.1272278, 3.1186566
7: -3.4549928, -0.3990459, -3.4729285, -0.3741739, -3.0227971, 3.0153401
8: 1.5520382, 3.8760381, 1.5346894, 3.8863626, -2.3169336, 2.2470584
9: -8.6979542, -5.0467758, -8.7047939, -5.0379648, -3.3748798, 3.3723154

Time for backsubstitution: 13.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5871
type: A, layer: 1, pos: 5871
type: B, layer: 1, pos: 4571
type: A, layer: 1, pos: 4571
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 5856
type: A, layer: 1, pos: 4682
type: B, layer: 1, pos: 4572
type: A, layer: 1, pos: 4572
type: B, layer: 1, pos: 5801
type: A, layer: 1, pos: 5801
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 5856
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 6139
type: B, layer: 1, pos: 4682
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 887
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 565

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5871

## Relational analysis of IS_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5871

## Relational analysis of IS_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4571

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6946206, upper bound: 1.7131144
time: 5.51 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7062601, upper bound: 1.7186147
time: 5.47 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -5.9742947, -2.1645575, -5.9743161, -2.1645081, -3.4510131, 3.3940849
1: -6.7698884, -3.7523479, -6.7699037, -3.7523165, -2.6996603, 2.7398188
2: -5.1139874, -2.1622519, -5.1139908, -2.1622114, -2.6749277, 2.6682427
3: -8.5303774, -4.5934467, -8.5304193, -4.5927138, -3.1535873, 3.1335917
4: -12.3269358, -8.3508997, -12.3269863, -8.3508720, -3.9654884, 3.9760866
5: -6.8727131, -3.7588899, -6.8727245, -3.7588861, -3.1138270, 3.0888681
6: -10.9138451, -7.3507900, -10.9138489, -7.3507247, -3.1397829, 3.1237652
7: -3.4867496, -0.3735521, -3.4868026, -0.3735511, -3.0469809, 3.0391281
8: 1.5337949, 3.8952460, 1.5337815, 3.8952699, -2.3338654, 2.2503107
9: -8.7087059, -5.0371356, -8.7087612, -5.0371342, -3.3834457, 3.3860645

Time for backsubstitution: 13.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5871
type: B, layer: 1, pos: 5871
type: B, layer: 1, pos: 4571
type: A, layer: 1, pos: 4571
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 5856
type: A, layer: 1, pos: 4682
type: A, layer: 1, pos: 4572
type: B, layer: 1, pos: 4572
type: B, layer: 1, pos: 5801
type: A, layer: 1, pos: 5801
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 5856
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 6139
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 4682
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 4573
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 565

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5871

## Relational analysis of IS_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5871

## Relational analysis of IS_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4571

## Relational analysis of IS_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7069607, upper bound: 1.7131144
time: 5.30 seconds

## Relational analysis of IS_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7186126, upper bound: 1.7186147
time: 5.48 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 36.17 seconds
IS_A1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 36.17
Output dim: 7, lower bound: -1.6962057, upper bound: 1.7069620
IS_A1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 36.17
Output dim: 7, lower bound: -1.7016701, upper bound: 1.7186141
IS_A1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 36.17
Output dim: 7, lower bound: -1.7085495, upper bound: 1.7069619
IS_A1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 36.17
Output dim: 7, lower bound: -1.7140274, upper bound: 1.7186131
IS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 36.17
Output dim: 7, lower bound: -1.6946206, upper bound: 1.7131144
IS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 36.17
Output dim: 7, lower bound: -1.7062601, upper bound: 1.7186147
IS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 36.17
Output dim: 7, lower bound: -1.7069607, upper bound: 1.7131144
IS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 36.17
Output dim: 7, lower bound: -1.7186126, upper bound: 1.7186147

## BFS IS instance: IS_A1_A1_A1

### Backsubstitution after applying IS history:
0: -5.8887682, -2.2852020, -5.9585886, -2.2134190, -3.2334337, 3.3303852
1: -6.6461353, -3.8244746, -6.7207413, -3.7627709, -2.6193819, 2.6172621
2: -5.0465093, -2.2282624, -5.1023898, -2.1868880, -2.5809569, 2.6096807
3: -8.4479847, -4.6735821, -8.5170555, -4.6281939, -3.0579515, 3.0706625
4: -12.1995897, -8.4411774, -12.2696495, -8.3655453, -3.8084888, 3.8284721
5: -6.8196306, -3.8151677, -6.8592062, -3.7769260, -3.0148287, 2.8791013
6: -10.8088522, -7.4641309, -10.8769140, -7.3833542, -2.9726305, 2.9380116
7: -3.3610425, -0.4585662, -3.4425764, -0.3916690, -2.8902540, 2.9274690
8: 1.6178632, 3.7367496, 1.5442762, 3.8270841, -2.1969852, 2.1379552
9: -8.6247301, -5.0731878, -8.6865625, -5.0464725, -3.2488174, 3.3156238

Time for backsubstitution: 13.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5871
type: B, layer: 1, pos: 5871
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 5801
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 4572
type: B, layer: 1, pos: 5856
type: B, layer: 1, pos: 4572
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 4682
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 5856
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 4571
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 4573
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 4573
type: A, layer: 1, pos: 887
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 4682
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 67

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5871

## Relational analysis of IS_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5871

## Relational analysis of IS_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 565

## Relational analysis of IS_A1_A1_A1_B1

### Relational analysis result of IS_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6962057, upper bound: 1.7024095
time: 5.12 seconds

## Relational analysis of IS_A1_A1_A1_B2

### Relational analysis result of IS_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6962057, upper bound: 1.7069620
time: 5.51 seconds

## BFS IS instance: IS_A1_A1_A2

### Backsubstitution after applying IS history:
0: -5.9056768, -2.2685003, -5.9658875, -2.2066183, -3.2668285, 3.3522701
1: -6.6940880, -3.7956381, -6.7428637, -3.7586756, -2.6954346, 2.6630425
2: -5.0652876, -2.2039108, -5.1073847, -2.1762409, -2.6134977, 2.6265173
3: -8.4888706, -4.6287403, -8.5236588, -4.6063290, -3.0930033, 3.0834417
4: -12.2359619, -8.4126215, -12.2873707, -8.3579817, -3.8779802, 3.8747492
5: -6.8329186, -3.7915709, -6.8631597, -3.7662508, -3.0418768, 3.0375900
6: -10.8734560, -7.4208088, -10.9074907, -7.3782434, -3.0098515, 3.0136337
7: -3.3961401, -0.4291317, -3.4524255, -0.3776696, -3.0069027, 2.9536850
8: 1.5867076, 3.7682586, 1.5381675, 3.8424721, -2.2089949, 2.1593611
9: -8.6516752, -5.0619197, -8.6946573, -5.0413761, -3.3187513, 3.3400035

Time for backsubstitution: 13.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5871
type: B, layer: 1, pos: 5871
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 5801
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 4572
type: A, layer: 1, pos: 4572
type: B, layer: 1, pos: 5856
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 5856
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 4682
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 4571
type: A, layer: 1, pos: 4682
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 4573
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 887
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 6139
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 67

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 5871

## Relational analysis of IS_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5871

## Relational analysis of IS_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 565

## Relational analysis of IS_A1_A1_A2_B1

### Relational analysis result of IS_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7016718, upper bound: 1.7140386
time: 5.17 seconds

## Relational analysis of IS_A1_A1_A2_B2

### Relational analysis result of IS_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7016701, upper bound: 1.7186141
time: 5.29 seconds

## BFS IS instance: IS_A1_A2_A1

### Backsubstitution after applying IS history:
0: -5.9027705, -2.2776971, -5.9600067, -2.2101841, -3.2447128, 3.3331308
1: -6.6535830, -3.8124890, -6.7217779, -3.7584319, -2.6328411, 2.6279006
2: -5.0642934, -2.2158737, -5.1040802, -2.1812651, -2.6049471, 2.6134090
3: -8.4518223, -4.6726613, -8.5185471, -4.6279926, -3.0611391, 3.0768461
4: -12.2196074, -8.4204979, -12.2792845, -8.3643360, -3.8259268, 3.8587866
5: -6.8257408, -3.8106573, -6.8615270, -3.7765634, -3.0226564, 2.8833971
6: -10.8180847, -7.4593349, -10.8781767, -7.3812113, -2.9849086, 2.9412568
7: -3.3923311, -0.4331405, -3.4564712, -0.3910470, -2.9092159, 2.9529641
8: 1.5996127, 3.7556610, 1.5433674, 3.8360009, -2.2138042, 2.1392596
9: -8.6349277, -5.0636787, -8.6905193, -5.0456510, -3.2538147, 3.3292098

Time for backsubstitution: 13.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5871
type: B, layer: 1, pos: 5871
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 5801
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 4572
type: B, layer: 1, pos: 5856
type: B, layer: 1, pos: 4572
type: A, layer: 1, pos: 4682
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 5856
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 4571
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 4573
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 4573
type: A, layer: 1, pos: 887
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 4682
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 67

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 5871

## Relational analysis of IS_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5871

## Relational analysis of IS_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 565

## Relational analysis of IS_A1_A2_A1_B1

### Relational analysis result of IS_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7085495, upper bound: 1.7024094
time: 5.44 seconds

## Relational analysis of IS_A1_A2_A1_B2

### Relational analysis result of IS_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7085495, upper bound: 1.7069619
time: 5.29 seconds

## BFS IS instance: IS_A1_A2_A2

### Backsubstitution after applying IS history:
0: -5.9196835, -2.2610102, -5.9673071, -2.2033842, -3.2781076, 3.3550000
1: -6.7015123, -3.7836576, -6.7438965, -3.7543347, -2.7088890, 2.6716294
2: -5.0831065, -2.1915102, -5.1090851, -2.1706161, -2.6375275, 2.6302514
3: -8.4925861, -4.6278214, -8.5251570, -4.6061258, -3.1014452, 3.0896344
4: -12.2560539, -8.3919487, -12.2970066, -8.3567734, -3.8992805, 3.9044676
5: -6.8390374, -3.7870221, -6.8654804, -3.7658846, -3.0496297, 3.0419145
6: -10.8827438, -7.4160185, -10.9087582, -7.3761001, -3.0220528, 3.0168731
7: -3.4274440, -0.4036636, -3.4663157, -0.3770428, -3.0201683, 2.9789355
8: 1.5684471, 3.7871747, 1.5372558, 3.8513889, -2.2258017, 2.1606767
9: -8.6618586, -5.0524139, -8.6986151, -5.0405588, -3.3237438, 3.3535943

Time for backsubstitution: 13.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5871
type: B, layer: 1, pos: 5871
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 5801
type: B, layer: 1, pos: 5801
type: A, layer: 1, pos: 4572
type: B, layer: 1, pos: 4572
type: B, layer: 1, pos: 5856
type: A, layer: 1, pos: 5856
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 4682
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 4682
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 4571
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 4573
type: A, layer: 1, pos: 4573
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 6139
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 67

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5871

## Relational analysis of IS_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5871

## Relational analysis of IS_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 565

## Relational analysis of IS_A1_A2_A2_B1

### Relational analysis result of IS_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7140274, upper bound: 1.7140372
time: 5.16 seconds

## Relational analysis of IS_A1_A2_A2_B2

### Relational analysis result of IS_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7140274, upper bound: 1.7186131
time: 5.18 seconds

## BFS IS instance: IS_A2_A1_B1

### Backsubstitution after applying IS history:
0: -5.9521828, -2.1788230, -5.9554300, -2.1843896, -3.4169178, 3.3643560
1: -6.7396851, -3.7684555, -6.7204361, -3.7855217, -2.6354718, 2.6505384
2: -5.0909944, -2.1851683, -5.0931921, -2.1922843, -2.6354771, 2.6312909
3: -8.5199451, -4.6165862, -8.4882050, -4.6380916, -3.0977831, 3.0887861
4: -12.2880383, -8.3792553, -12.2799225, -8.3806629, -3.8870068, 3.8538046
5: -6.8624091, -3.7745154, -6.8569756, -3.7831874, -2.9540415, 3.0583730
6: -10.8736963, -7.3610377, -10.8477764, -7.3964729, -3.0509014, 3.0252914
7: -3.4451504, -0.4130592, -3.4373899, -0.4037192, -2.9766169, 2.9118433
8: 1.5581756, 3.8606339, 1.5659876, 3.8545198, -2.2766879, 2.2250772
9: -8.6898432, -5.0518761, -8.6773367, -5.0492420, -3.3512449, 3.3161736

Time for backsubstitution: 13.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5871
type: A, layer: 1, pos: 5871
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 5858
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 5856
type: B, layer: 1, pos: 4572
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 5801
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 4682
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 4682
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 4571
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 4573
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 4573
type: B, layer: 1, pos: 5856
type: A, layer: 1, pos: 6139
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 887
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 5858

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5871

## Relational analysis of IS_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5871

## Relational analysis of IS_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4666

## Relational analysis of IS_A2_A1_B1_A1

### Relational analysis result of IS_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6925316, upper bound: 1.7125853
time: 5.08 seconds

## Relational analysis of IS_A2_A1_B1_A2

### Relational analysis result of IS_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6946114, upper bound: 1.7131056
time: 5.81 seconds

## BFS IS instance: IS_A2_A1_B2

### Backsubstitution after applying IS history:
0: -5.9594955, -2.1720290, -5.9728546, -2.1677654, -3.4392223, 3.3913512
1: -6.7618036, -3.7643533, -6.7687941, -3.7566674, -2.6866350, 2.7049761
2: -5.0960021, -2.1745062, -5.1122985, -2.1677999, -2.6522398, 2.6644459
3: -8.5265989, -4.5947323, -8.5288811, -4.5929976, -3.1127806, 3.1293015
4: -12.3058167, -8.3716755, -12.3172932, -8.3521023, -3.9353943, 3.9448233
5: -6.8663793, -3.7638540, -6.8703756, -3.7592883, -3.1070910, 3.0853233
6: -10.9042635, -7.3559289, -10.9124584, -7.3528786, -3.1172695, 3.0621433
7: -3.4549832, -0.3990622, -3.4729066, -0.3742092, -3.0028906, 3.0049002
8: 1.5520439, 3.8760200, 1.5347018, 3.8863211, -2.2955863, 2.2470279
9: -8.6979446, -5.0467806, -8.7047749, -5.0379758, -3.3756371, 3.3722854

Time for backsubstitution: 13.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5871
type: A, layer: 1, pos: 5871
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 5856
type: A, layer: 1, pos: 4682
type: B, layer: 1, pos: 4572
type: A, layer: 1, pos: 4572
type: B, layer: 1, pos: 5801
type: A, layer: 1, pos: 5801
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 4571
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 5856
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 4682
type: B, layer: 1, pos: 4573
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 4573
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 565

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5871

## Relational analysis of IS_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5871

## Relational analysis of IS_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5858

## Relational analysis of IS_A2_A1_B2_A1

### Relational analysis result of IS_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7042912, upper bound: 1.7133801
time: 5.40 seconds

## Relational analysis of IS_A2_A1_B2_A2

### Relational analysis result of IS_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7062584, upper bound: 1.7186141
time: 5.19 seconds

## BFS IS instance: IS_A2_A2_B1

### Backsubstitution after applying IS history:
0: -5.9669709, -2.1713626, -5.9568739, -2.1811538, -3.4288096, 3.3670449
1: -6.7477388, -3.7564564, -6.7214890, -3.7811716, -2.6485009, 2.6611474
2: -5.1089540, -2.1729350, -5.0948548, -2.1867242, -2.6543264, 2.6350334
3: -8.5237589, -4.6153374, -8.4897194, -4.6378860, -3.1010256, 3.0930133
4: -12.3091230, -8.3584824, -12.2895575, -8.3794470, -3.9089460, 3.8720026
5: -6.8687325, -3.7695820, -6.8593197, -3.7828269, -2.9626150, 3.0626154
6: -10.8832169, -7.3558989, -10.8490686, -7.3943281, -3.0634537, 3.0303850
7: -3.4769039, -0.3875821, -3.4512815, -0.4031055, -3.0010848, 2.9356062
8: 1.5399394, 3.8798394, 1.5650878, 3.8634272, -2.2935779, 2.2282016
9: -8.7005911, -5.0422354, -8.6813040, -5.0484104, -3.3598080, 3.3299184

Time for backsubstitution: 13.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5871
type: B, layer: 1, pos: 5871
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 5858
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 5856
type: B, layer: 1, pos: 4572
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 5801
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 4682
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 4682
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 4571
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 5856
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 6139
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 887
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 5858

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5871

## Relational analysis of IS_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5871

## Relational analysis of IS_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4666

## Relational analysis of IS_A2_A2_B1_A1

### Relational analysis result of IS_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7048677, upper bound: 1.7125843
time: 5.64 seconds

## Relational analysis of IS_A2_A2_B1_A2

### Relational analysis result of IS_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7069515, upper bound: 1.7131056
time: 5.30 seconds

## BFS IS instance: IS_A2_A2_B2

### Backsubstitution after applying IS history:
0: -5.9742861, -2.1645684, -5.9743013, -2.1645284, -3.4511094, 3.3940511
1: -6.7698560, -3.7523518, -6.7698312, -3.7523258, -2.6996174, 2.7149248
2: -5.1139827, -2.1622627, -5.1139779, -2.1622367, -2.6705551, 2.6682141
3: -8.5303688, -4.5934811, -8.5304050, -4.5927896, -3.1159830, 3.1335368
4: -12.3269119, -8.3509073, -12.3269310, -8.3508892, -3.9573431, 3.9630365
5: -6.8727088, -3.7589035, -6.8727169, -3.7589166, -3.1137922, 3.0897164
6: -10.9138012, -7.3507953, -10.9137583, -7.3507385, -3.1265283, 3.0672534
7: -3.4867387, -0.3735685, -3.4867806, -0.3735852, -3.0270205, 3.0286880
8: 1.5338011, 3.8952284, 1.5337944, 3.8952289, -2.3125272, 2.2491267
9: -8.7086964, -5.0371399, -8.7087421, -5.0371442, -3.3842039, 3.3860354

Time for backsubstitution: 13.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5871
type: B, layer: 1, pos: 5871
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 5856
type: A, layer: 1, pos: 4682
type: A, layer: 1, pos: 4572
type: B, layer: 1, pos: 4572
type: B, layer: 1, pos: 5801
type: A, layer: 1, pos: 5801
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 4571
type: B, layer: 1, pos: 5856
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 4682
type: B, layer: 1, pos: 4573
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 4573
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 6139
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 565

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5871

## Relational analysis of IS_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5871

## Relational analysis of IS_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5858

## Relational analysis of IS_A2_A2_B2_A1

### Relational analysis result of IS_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7167911, upper bound: 1.7133801
time: 5.27 seconds

## Relational analysis of IS_A2_A2_B2_A2

### Relational analysis result of IS_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7186121, upper bound: 1.7186141
time: 5.08 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 36.02 seconds
IS_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 36.02
Output dim: 7, lower bound: -1.6962057, upper bound: 1.7024095
IS_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 36.02
Output dim: 7, lower bound: -1.6962057, upper bound: 1.7069620
IS_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 36.02
Output dim: 7, lower bound: -1.7016718, upper bound: 1.7140386
IS_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 36.02
Output dim: 7, lower bound: -1.7016701, upper bound: 1.7186141
IS_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 36.02
Output dim: 7, lower bound: -1.7085495, upper bound: 1.7024094
IS_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 36.02
Output dim: 7, lower bound: -1.7085495, upper bound: 1.7069619
IS_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 36.02
Output dim: 7, lower bound: -1.7140274, upper bound: 1.7140372
IS_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 36.02
Output dim: 7, lower bound: -1.7140274, upper bound: 1.7186131
IS_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 36.02
Output dim: 7, lower bound: -1.6925316, upper bound: 1.7125853
IS_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 36.02
Output dim: 7, lower bound: -1.6946114, upper bound: 1.7131056
IS_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 36.02
Output dim: 7, lower bound: -1.7042912, upper bound: 1.7133801
IS_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 36.02
Output dim: 7, lower bound: -1.7062584, upper bound: 1.7186141
IS_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 36.02
Output dim: 7, lower bound: -1.7048677, upper bound: 1.7125843
IS_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 36.02
Output dim: 7, lower bound: -1.7069515, upper bound: 1.7131056
IS_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 36.02
Output dim: 7, lower bound: -1.7167911, upper bound: 1.7133801
IS_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 36.02
Output dim: 7, lower bound: -1.7186121, upper bound: 1.7186141

## BFS IS instance: IS_A1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -5.8887682, -2.2852020, -5.9112577, -2.2710276, -3.1958165, 3.2074013
1: -6.6461353, -3.8244746, -6.6786451, -3.7920661, -2.5851269, 2.6360471
2: -5.0465093, -2.2282624, -5.0765200, -2.2078233, -2.5533366, 2.5750358
3: -8.4479847, -4.6735821, -8.4844980, -4.6489840, -3.0459909, 3.0378861
4: -12.1995897, -8.4411774, -12.2290907, -8.4006748, -3.7745266, 3.7879133
5: -6.8196306, -3.8151677, -6.8328218, -3.7978961, -2.9786644, 2.8474822
6: -10.8088522, -7.4641309, -10.8509979, -7.4230704, -2.8885269, 2.9080224
7: -3.3610425, -0.4585662, -3.4038863, -0.4182589, -2.8604236, 2.9235749
8: 1.6178632, 3.7367496, 1.5754118, 3.7630420, -2.1050673, 2.1019363
9: -8.6247301, -5.0731878, -8.6501427, -5.0583062, -3.2347984, 3.2778273

Time for backsubstitution: 13.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5871
type: A, layer: 1, pos: 5871
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 5801
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 4572
type: B, layer: 1, pos: 4572
type: B, layer: 1, pos: 5856
type: A, layer: 1, pos: 4682
type: A, layer: 1, pos: 5858
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 5856
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 4571
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 4573
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 4573
type: A, layer: 1, pos: 887
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 4682
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 6139
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 67

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5871

## Relational analysis of IS_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5871

## Relational analysis of IS_A1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4666

## Relational analysis of IS_A1_A1_A1_B1_B1

### Relational analysis result of IS_A1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6956724, upper bound: 1.7003201
time: 4.96 seconds

## Relational analysis of IS_A1_A1_A1_B1_B2

### Relational analysis result of IS_A1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6961965, upper bound: 1.7024003
time: 5.20 seconds

## BFS IS instance: IS_A1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -5.8887682, -2.2852020, -5.9655461, -2.1745696, -3.2362423, 3.3289661
1: -6.6461353, -3.8244746, -6.7466998, -3.7607684, -2.6227398, 2.6249399
2: -5.0465093, -2.2282624, -5.1072822, -2.1784573, -2.5783920, 2.6139615
3: -8.4479847, -4.6735821, -8.5222788, -4.6148181, -3.0671258, 3.0755210
4: -12.1995897, -8.4411774, -12.2995224, -8.3596745, -3.8086128, 3.8535023
5: -6.8196306, -3.8151677, -6.8664017, -3.7699475, -3.0172610, 2.8785386
6: -10.8088522, -7.4641309, -10.8819227, -7.3579869, -2.9981036, 2.9432883
7: -3.3610425, -0.4585662, -3.4630675, -0.3882012, -2.8823400, 2.9392254
8: 1.6178632, 3.7367496, 1.5408320, 3.8709388, -2.2054744, 2.1354580
9: -8.6247301, -5.0731878, -8.6966801, -5.0430627, -3.2504635, 3.3260260

Time for backsubstitution: 13.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5871
type: A, layer: 1, pos: 5871
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 5801
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 4682
type: B, layer: 1, pos: 4572
type: A, layer: 1, pos: 5856
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 5856
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 4571
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 4573
type: A, layer: 1, pos: 887
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 6139
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 4682
type: B, layer: 1, pos: 5858
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 67

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5871

## Relational analysis of IS_A1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5871

## Relational analysis of IS_A1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4666

## Relational analysis of IS_A1_A1_A1_B2_B1

### Relational analysis result of IS_A1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6956724, upper bound: 1.7048679
time: 5.04 seconds

## Relational analysis of IS_A1_A1_A1_B2_B2

### Relational analysis result of IS_A1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6961965, upper bound: 1.7069529
time: 5.41 seconds

## BFS IS instance: IS_A1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -5.9056768, -2.2685003, -5.9183211, -2.2642093, -3.2229724, 3.2311702
1: -6.6940880, -3.7956381, -6.7005286, -3.7879753, -2.6611290, 2.6862493
2: -5.0652876, -2.2039108, -5.0813823, -2.1972177, -2.5858274, 2.5938551
3: -8.4888706, -4.6287403, -8.4911242, -4.6272449, -3.0892053, 3.0506907
4: -12.2359619, -8.4126215, -12.2464809, -8.3931122, -3.8428497, 3.8338594
5: -6.8329186, -3.7915709, -6.8367410, -3.7873652, -3.0051680, 3.0059767
6: -10.8734560, -7.4208088, -10.8815727, -7.4181027, -2.9219236, 2.9836535
7: -3.3961401, -0.4291317, -3.4135702, -0.4042742, -2.9770651, 2.9497023
8: 1.5867076, 3.7682586, 1.5693488, 3.7782764, -2.1320829, 2.1232841
9: -8.6516752, -5.0619197, -8.6579781, -5.0532112, -3.3047295, 3.3026252

Time for backsubstitution: 13.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5871
type: A, layer: 1, pos: 5871
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 5801
type: A, layer: 1, pos: 5801
type: B, layer: 1, pos: 4572
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 5856
type: B, layer: 1, pos: 5856
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 5858
type: A, layer: 1, pos: 5858
type: B, layer: 1, pos: 4682
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4682
type: B, layer: 1, pos: 4571
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 4573
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 887
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 6139
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 67

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5871

## Relational analysis of IS_A1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5871

## Relational analysis of IS_A1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4666

## Relational analysis of IS_A1_A1_A2_B1_A1

### Relational analysis result of IS_A1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6995718, upper bound: 1.7134947
time: 7.21 seconds

## Relational analysis of IS_A1_A1_A2_B1_A2

### Relational analysis result of IS_A1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7016607, upper bound: 1.7140293
time: 5.16 seconds

## BFS IS instance: IS_A1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -5.9056768, -2.2685003, -5.9728608, -2.1677742, -3.2696342, 3.3513465
1: -6.6940880, -3.7956381, -6.7688189, -3.7566638, -2.6988034, 2.6665497
2: -5.0652876, -2.2039108, -5.1123042, -2.1677885, -2.6109681, 2.6307216
3: -8.4888706, -4.6287403, -8.5288868, -4.5929618, -3.0957460, 3.0883608
4: -12.2359619, -8.4126215, -12.3173094, -8.3520975, -3.8838644, 3.8977451
5: -6.8329186, -3.7915709, -6.8703780, -3.7592735, -3.0443516, 3.0370464
6: -10.8734560, -7.4208088, -10.9125013, -7.3528814, -3.0353608, 3.0109942
7: -3.3961401, -0.4291317, -3.4729099, -0.3741915, -2.9933681, 2.9651940
8: 1.5867076, 3.7682586, 1.5346956, 3.8863273, -2.2175753, 2.1527863
9: -8.6516752, -5.0619197, -8.7047844, -5.0379715, -3.3203983, 3.3504148

Time for backsubstitution: 13.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5871
type: A, layer: 1, pos: 5871
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 5801
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 4572
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 5856
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 5856
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 4682
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 4571
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 4682
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 4573
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 5858
type: A, layer: 1, pos: 887
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 6139
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 67

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5871

## Relational analysis of IS_A1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5871

## Relational analysis of IS_A1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4666

## Relational analysis of IS_A1_A1_A2_B2_A1

### Relational analysis result of IS_A1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6995718, upper bound: 1.7180727
time: 6.39 seconds

## Relational analysis of IS_A1_A1_A2_B2_A2

### Relational analysis result of IS_A1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7016607, upper bound: 1.7186049
time: 5.06 seconds

## BFS IS instance: IS_A1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -5.9027705, -2.2776971, -5.9126477, -2.2677855, -3.2134361, 3.2127347
1: -6.6535830, -3.8124890, -6.6796708, -3.7877147, -2.5985918, 2.6487281
2: -5.0642934, -2.2158737, -5.0782447, -2.2020657, -2.5772963, 2.5794613
3: -8.4518223, -4.6726613, -8.4860067, -4.6487885, -3.0489855, 3.0440845
4: -12.2196074, -8.4204979, -12.2387266, -8.3994761, -3.7919521, 3.8182287
5: -6.8257408, -3.8106573, -6.8351326, -3.7975423, -2.9856014, 2.8517437
6: -10.8180847, -7.4593349, -10.8522110, -7.4209270, -2.9007182, 2.9112144
7: -3.3923311, -0.4331405, -3.4178195, -0.4176352, -2.8792768, 2.9529424
8: 1.5996127, 3.7556610, 1.5744953, 3.7719626, -2.1311276, 2.1032224
9: -8.6349277, -5.0636787, -8.6540823, -5.0575018, -3.2397738, 3.2912579

Time for backsubstitution: 13.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5871
type: B, layer: 1, pos: 5871
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 5801
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 4572
type: B, layer: 1, pos: 5856
type: B, layer: 1, pos: 4572
type: A, layer: 1, pos: 4682
type: A, layer: 1, pos: 5858
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 5856
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 4571
type: B, layer: 1, pos: 5858
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 4573
type: A, layer: 1, pos: 887
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 4682
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 6139
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 67

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5871

## Relational analysis of IS_A1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5871

## Relational analysis of IS_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4666

## Relational analysis of IS_A1_A2_A1_B1_B1

### Relational analysis result of IS_A1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7080189, upper bound: 1.7003211
time: 5.53 seconds

## Relational analysis of IS_A1_A2_A1_B1_B2

### Relational analysis result of IS_A1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7085405, upper bound: 1.7024002
time: 5.07 seconds

## BFS IS instance: IS_A1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -5.9027705, -2.2776971, -5.9669895, -2.1713321, -3.2475119, 3.3295333
1: -6.6535830, -3.8124890, -6.7477393, -3.7564249, -2.6362019, 2.6335373
2: -5.0642934, -2.2158737, -5.1089549, -2.1728954, -2.6021795, 2.6176939
3: -8.4518223, -4.6726613, -8.5238008, -4.6146131, -3.0749216, 3.0817208
4: -12.2196074, -8.4204979, -12.3091593, -8.3584595, -3.8179770, 3.8716116
5: -6.8257408, -3.8106573, -6.8687429, -3.7695818, -3.0250983, 2.8828669
6: -10.8180847, -7.4593349, -10.8832169, -7.3558435, -3.0103765, 2.9465747
7: -3.3923311, -0.4331405, -3.4769466, -0.3875809, -2.8948245, 2.9629967
8: 1.5996127, 3.7556610, 1.5399284, 3.8798466, -2.2222810, 2.1341016
9: -8.6349277, -5.0636787, -8.7006435, -5.0422330, -3.2554722, 3.3396258

Time for backsubstitution: 13.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5871
type: B, layer: 1, pos: 5871
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 5801
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 4682
type: B, layer: 1, pos: 4572
type: A, layer: 1, pos: 5856
type: B, layer: 1, pos: 5856
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 4571
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 4573
type: A, layer: 1, pos: 887
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 5858
type: A, layer: 1, pos: 6139
type: B, layer: 1, pos: 4682
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 67

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5871

## Relational analysis of IS_A1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5871

## Relational analysis of IS_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4666

## Relational analysis of IS_A1_A2_A1_B2_B1

### Relational analysis result of IS_A1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7080189, upper bound: 1.7048688
time: 5.60 seconds

## Relational analysis of IS_A1_A2_A1_B2_B2

### Relational analysis result of IS_A1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7085405, upper bound: 1.7069527
time: 5.34 seconds

## BFS IS instance: IS_A1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -5.9196835, -2.2610102, -5.9197092, -2.2609682, -3.2405882, 3.2364855
1: -6.7015123, -3.7836576, -6.7015519, -3.7836227, -2.6745858, 2.6989255
2: -5.0831065, -2.1915102, -5.0831165, -2.1914577, -2.6098638, 2.5982983
3: -8.4925861, -4.6278214, -8.4926376, -4.6270514, -3.0921297, 3.0568981
4: -12.2560539, -8.3919487, -12.2561207, -8.3919144, -3.8641396, 3.8641720
5: -6.8390374, -3.7870221, -6.8390508, -3.7870066, -3.0120382, 3.0102673
6: -10.8827438, -7.4160185, -10.8827915, -7.4159594, -2.9340930, 2.9868400
7: -3.4274440, -0.4036636, -3.4274969, -0.4036460, -2.9902158, 2.9787765
8: 1.5684471, 3.7871747, 1.5684309, 3.7871985, -2.1489244, 2.1245799
9: -8.6618586, -5.0524139, -8.6619215, -5.0524073, -3.3097010, 3.3160596

Time for backsubstitution: 13.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5871
type: B, layer: 1, pos: 5871
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 5801
type: B, layer: 1, pos: 5801
type: A, layer: 1, pos: 4572
type: B, layer: 1, pos: 4572
type: B, layer: 1, pos: 5856
type: A, layer: 1, pos: 5856
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 5858
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 4682
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 4682
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 4571
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 4573
type: A, layer: 1, pos: 4573
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 6139
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 67

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5871

## Relational analysis of IS_A1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5871

## Relational analysis of IS_A1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4666

## Relational analysis of IS_A1_A2_A2_B1_A1

### Relational analysis result of IS_A1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7119349, upper bound: 1.7134945
time: 5.24 seconds

## Relational analysis of IS_A1_A2_A2_B1_A2

### Relational analysis result of IS_A1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.7140181, upper bound: 1.7140283
time: 5.41 seconds

## BFS IS instance: IS_A1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -5.9196835, -2.2610102, -5.9743052, -2.1645377, -3.2809043, 3.3519039
1: -6.7015123, -3.7836576, -6.7698574, -3.7523212, -2.7090955, 2.6751349
2: -5.0831065, -2.1915102, -5.1139846, -2.1622262, -2.6345935, 2.6344709
3: -8.4925861, -4.6278214, -8.5304098, -4.5927553, -3.1041861, 3.0945673
4: -12.2560539, -8.3919487, -12.3269463, -8.3508835, -3.9051704, 3.9158707
5: -6.8390374, -3.7870221, -6.8727188, -3.7589030, -3.0521135, 3.0414042
6: -10.8827438, -7.4160185, -10.9138021, -7.3507395, -3.0466266, 3.0128086
7: -3.4274440, -0.4036636, -3.4867835, -0.3735669, -3.0058479, 2.9889700
8: 1.5684471, 3.7871747, 1.5337906, 3.8952336, -2.2343693, 2.1514421
9: -8.6618586, -5.0524139, -8.7087498, -5.0371404, -3.3254023, 3.3640199

Time for backsubstitution: 13.84 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=3.0680713653564453
rel_dist={7: [-1.7205280069910254, 1.7205289191635718]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 5871
type: A, layer: 1, pos: 5871
type: B, layer: 1, pos: 4571
type: A, layer: 1, pos: 4571
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 4572
type: B, layer: 1, pos: 4572
type: A, layer: 1, pos: 5801
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 5856
type: A, layer: 1, pos: 5856
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 4682
type: A, layer: 1, pos: 4682
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 5858
type: A, layer: 1, pos: 5858
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 887
type: B, layer: 1, pos: 4573
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 67

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 565

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6196670, upper bound: 1.6159423
time: 5.54 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6196685, upper bound: 1.6196662
time: 4.97 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 10.74 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 10.74
Output dim: 7, lower bound: -1.6196670, upper bound: 1.6159423
IS_B2, status: Status.UNKNOWN, split count: 1, time: 10.74
Output dim: 7, lower bound: -1.6196685, upper bound: 1.6196662

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -5.9658685, -2.2113938, -5.9197569, -2.2608802, -3.3069091, 3.2228684
1: -6.7385573, -3.7547290, -6.7016020, -3.7835846, -2.6226888, 2.6761308
2: -5.1080465, -2.1721153, -5.0831227, -2.1913643, -2.5975890, 2.5943387
3: -8.5241251, -4.6082263, -8.4927025, -4.6263618, -3.0334735, 3.0122972
4: -12.2920322, -8.3579693, -12.2562580, -8.3918877, -3.8617449, 3.8830490
5: -6.8639884, -3.7675302, -6.8390732, -3.7869825, -2.9925833, 2.9949212
6: -10.9077282, -7.3813238, -10.8828382, -7.4158897, -2.9502649, 3.0038071
7: -3.4624531, -0.3777528, -3.4275479, -0.4036295, -2.9709902, 2.9984694
8: 1.5379558, 3.8422813, 1.5684128, 3.7872210, -2.1527328, 2.2028987
9: -8.6963110, -5.0412655, -8.6620216, -5.0523887, -3.2876129, 3.2622209

Time for backsubstitution: 13.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 5871
type: A, layer: 1, pos: 5871
type: B, layer: 1, pos: 4571
type: A, layer: 1, pos: 4571
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 5801
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 4572
type: B, layer: 1, pos: 4572
type: A, layer: 1, pos: 5856
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 5856
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 4682
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 4682
type: B, layer: 1, pos: 5858
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 4573
type: B, layer: 1, pos: 4573
type: A, layer: 1, pos: 887
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 6139
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 67

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 957

## Relational analysis of IS_B1_B1

### Relational analysis result of IS_B1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.6179044, upper bound: 1.6047514
time: 5.07 seconds

## Relational analysis of IS_B1_B2

### Relational analysis result of IS_B1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.6179044, upper bound: 1.6141844
time: 5.10 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -5.9743538, -2.1644320, -5.9743538, -2.1644485, -3.3441291, 3.4035242
1: -6.7699184, -3.7522845, -6.7699089, -3.7522840, -2.6888752, 2.6441123
2: -5.1139936, -2.1621275, -5.1139917, -2.1621308, -2.6322570, 2.6326256
3: -8.5304775, -4.5920610, -8.5304775, -4.5920672, -3.0453310, 3.0668802
4: -12.3270969, -8.3508511, -12.3270874, -8.3508549, -3.9340916, 3.9030170
5: -6.8727446, -3.7588763, -6.8727422, -3.7588780, -3.0366421, 3.0677156
6: -10.9138498, -7.3506641, -10.9138508, -7.3506694, -3.0552359, 3.0703452
7: -3.4868441, -0.3735499, -3.4868371, -0.3735507, -3.0216150, 3.0164847
8: 1.5337715, 3.8952708, 1.5337729, 3.8952570, -2.2181993, 2.3019729
9: -8.7088518, -5.0371184, -8.7088499, -5.0371222, -3.3216763, 3.3220019

Time for backsubstitution: 13.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 5871
type: B, layer: 1, pos: 5871
type: A, layer: 1, pos: 4571
type: B, layer: 1, pos: 4571
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 5858
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 5856
type: A, layer: 1, pos: 4572
type: B, layer: 1, pos: 4572
type: B, layer: 1, pos: 4682
type: A, layer: 1, pos: 5801
type: B, layer: 1, pos: 5801
type: A, layer: 1, pos: 4626
type: B, layer: 1, pos: 4626
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 6139
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 5856
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 4682
type: A, layer: 1, pos: 887
type: B, layer: 1, pos: 4573
type: A, layer: 1, pos: 4573
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 565

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 957

## Relational analysis of IS_B2_B1

### Relational analysis result of IS_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.6179044, upper bound: 1.6084815
time: 5.12 seconds

## Relational analysis of IS_B2_B2

### Relational analysis result of IS_B2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.6179044, upper bound: 1.6179027
time: 4.84 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 23.88 seconds
IS_B1_B1, status: Status.VERIFIED, split count: 2, time: 23.88
Output dim: 7, lower bound: -1.6179044, upper bound: 1.6047514
IS_B1_B2, status: Status.VERIFIED, split count: 2, time: 23.88
Output dim: 7, lower bound: -1.6179044, upper bound: 1.6141844
IS_B2_B1, status: Status.VERIFIED, split count: 2, time: 23.88
Output dim: 7, lower bound: -1.6179044, upper bound: 1.6084815
IS_B2_B2, status: Status.VERIFIED, split count: 2, time: 23.88
Output dim: 7, lower bound: -1.6179044, upper bound: 1.6179027
Binary search (step 2): status=Status.VERIFIED, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=3.029996871948242
rel_dist={7: [-1.6196747972199932, 1.6196725960818332]}

## Binary Search with IS_dual Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01171875
execution time: 1725.56 seconds
