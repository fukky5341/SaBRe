## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_1.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 3.05840151


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395)
1: (-1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596)
2: (-1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697)
3: (-3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084)
4: (-2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487)

## BASE Result
execution time: IAR + LP analysis = 2.14 + 1.27 = 3.42 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -3.3982887, upper bound: 3.3982887


# Binary Search by BASE starts (time budget: 1196.58 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.2500000


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=3.5238394737243652
rel_dist={0: [-3.3982802470538505, 3.398280247053849]}

## Binary search (step 1) starts
Candidate diff: 0.1250000


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=3.5238394737243652
rel_dist={0: [-3.3982329481346, 3.3982329481346003]}

## Binary search (step 2) starts
Candidate diff: 0.0625000


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=3.5238394737243652
rel_dist={0: [-3.398203369636585, 3.398203369636585]}

## Binary search (step 3) starts
Candidate diff: 0.0312500


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0312500, mid=0.0312500, abs_max=3.5238394737243652
rel_dist={0: [-3.39818007524518, 3.398180075245179]}

## Binary search (step 4) starts
Candidate diff: 0.0156250


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0156250, mid=0.0156250, abs_max=3.5238394737243652
rel_dist={0: [-3.3981656742023487, 3.3981656742023487]}

## Binary search (step 5) starts
Candidate diff: 0.0078125


## IAR start
Binary search (step 5): status=Status.UNKNOWN, low=0.0000000, high=0.0078125, mid=0.0078125, abs_max=3.5238394737243652
rel_dist={0: [-3.3981560004933664, 3.3981560004933664]}

## Binary search (step 6) starts
Candidate diff: 0.0039062


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0000000, high=0.0039062, mid=0.0039062, abs_max=3.5238394737243652
rel_dist={0: [-3.398151163638105, 3.398151163638106]}

## Binary search (step 7) starts
Candidate diff: 0.0019531


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0000000, high=0.0019531, mid=0.0019531, abs_max=3.5238394737243652
rel_dist={0: [-3.398148745209495, 3.398148745209495]}

## Binary search (step 8) starts
Candidate diff: 0.0009766


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0000000, high=0.0009766, mid=0.0009766, abs_max=3.5238394737243652
rel_dist={0: [-3.398147442234788, 3.398147442234788]}

## Binary search (step 9) starts
Candidate diff: 0.0004883


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0000000, high=0.0004883, mid=0.0004883, abs_max=3.5238394737243652
rel_dist={0: [-3.3981467631830444, 3.3981467631830444]}

## Binary search (step 10) starts
Candidate diff: 0.0002441


## IAR start
Binary search (step 10): status=Status.UNKNOWN, low=0.0000000, high=0.0002441, mid=0.0002441, abs_max=3.5238394737243652
rel_dist={0: [-3.3981464236572503, 3.3981464236572503]}

## Binary search (step 11) starts
Candidate diff: 0.0001221


## IAR start
Binary search (step 11): status=Status.UNKNOWN, low=0.0000000, high=0.0001221, mid=0.0001221, abs_max=3.5238394737243652
rel_dist={0: [-3.398146253894503, 3.3981462538945024]}

## Binary search (step 12) starts
Candidate diff: 0.0000610


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0000000, high=0.0000610, mid=0.0000610, abs_max=3.5238394737243652
rel_dist={0: [-3.3981461690134305, 3.3981461690134305]}

## Binary search (step 13) starts
Candidate diff: 0.0000305


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0000000, high=0.0000305, mid=0.0000305, abs_max=3.5238394737243652
rel_dist={0: [-3.398146126573492, 3.398146126573492]}

## Binary search (step 14) starts
Candidate diff: 0.0000153


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0000000, high=0.0000153, mid=0.0000153, abs_max=3.5238394737243652
rel_dist={0: [-3.3981461053547006, 3.3981461053547006]}

## Binary search (step 15) starts
Candidate diff: 0.0000076


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0000000, high=0.0000076, mid=0.0000076, abs_max=3.5238394737243652
rel_dist={0: [-3.398146094747597, 3.398146094747597]}

## Binary search (step 16) starts
Candidate diff: 0.0000038


## IAR start
Binary search (step 16): status=Status.UNKNOWN, low=0.0000000, high=0.0000038, mid=0.0000038, abs_max=3.5238394737243652
rel_dist={0: [-3.3981460894483844, 3.3981460894483844]}

## Binary search (step 17) starts
Candidate diff: 0.0000019


## IAR start
Binary search (step 17): status=Status.UNKNOWN, low=0.0000000, high=0.0000019, mid=0.0000019, abs_max=3.5238394737243652
rel_dist={0: [-3.3981460896207114, 3.3981460868065954]}

## Binary search (step 18) starts
Candidate diff: 0.0000010


## IAR start
Binary search (step 18): status=Status.UNKNOWN, low=0.0000000, high=0.0000010, mid=0.0000010, abs_max=3.5238394737243652
rel_dist={0: [-3.3981461239315784, 3.3981460938166226]}

## Binary Search Result
Binary search time: 65.92 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 1130.66 seconds

## Binary search (step 0) starts
Candidate diff: 0.2500000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3925085, upper bound: 3.3805396
time: 0.41 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3966881, upper bound: 3.3966880
time: 0.39 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.00 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.00
Output dim: 0, lower bound: -3.3925085, upper bound: 3.3805396
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.00
Output dim: 0, lower bound: -3.3966881, upper bound: 3.3966880

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.8476034, 1.4167995, -1.2466472, 2.2771921, -3.1247954, 2.6634469
1: -1.3425418, 1.9780073, -1.9637374, 3.1950235, -4.5375633, 3.9417448
2: -0.9108999, 2.0947752, -1.3604455, 3.2710245, -4.1819229, 3.4552207
3: -2.3312497, 2.5655317, -3.4595599, 4.0564485, -6.3876977, 6.0250916
4: -1.4425111, 2.7361240, -2.1785955, 4.2066536, -5.6491647, 4.9147182

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3885635, upper bound: 3.3609978
time: 0.40 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3511452, upper bound: 3.3526375
time: 0.41 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.9148694, 1.5653143, -1.2094771, 2.1952481, -3.1101174, 2.7747912
1: -1.4585052, 2.2524974, -1.9063909, 3.0850666, -4.5435715, 4.1588879
2: -0.9964871, 2.2692239, -1.3182706, 3.1591482, -4.1556354, 3.5874944
3: -2.5034299, 2.9087937, -3.3527398, 3.9233556, -6.4267855, 6.2615328
4: -1.5905116, 2.9554746, -2.1098680, 4.0681944, -5.6587057, 5.0653424

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3881330, upper bound: 3.3590412
time: 0.40 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3506071, upper bound: 3.3506071
time: 0.44 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.14 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.14
Output dim: 0, lower bound: -3.3885635, upper bound: 3.3609978
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.14
Output dim: 0, lower bound: -3.3511452, upper bound: 3.3526375
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.14
Output dim: 0, lower bound: -3.3881330, upper bound: 3.3590412
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.14
Output dim: 0, lower bound: -3.3506071, upper bound: 3.3506071

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.8288031, 1.3775676, -1.4205828, 2.6549139, -3.4837165, 2.7981505
1: -1.3134019, 1.9227006, -2.2293470, 3.7171719, -5.0305738, 4.1520472
2: -0.8896832, 2.0402029, -1.5606055, 3.8029828, -4.6926661, 3.6008079
3: -2.2793450, 2.4964702, -3.9579871, 4.6879396, -6.9672847, 6.4544573
4: -1.4078611, 2.6674209, -2.5043542, 4.8521762, -6.2600360, 5.1717749

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 37

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3821391, upper bound: 3.3444861
time: 0.44 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3877086, upper bound: 3.3609976
time: 0.44 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.8476034, 1.4167995, -1.0157527, 1.7855062, -2.6331091, 2.4325523
1: -1.3425418, 1.9780073, -1.6081196, 2.5250821, -3.8676236, 3.5861268
2: -0.9108999, 2.0947752, -1.0928304, 2.5890822, -3.4999819, 3.1876056
3: -2.3312497, 2.5655317, -2.8151433, 3.2343123, -5.5655622, 5.3806753
4: -1.4425111, 2.7361240, -1.7422895, 3.3683147, -4.8108258, 4.4784136

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3456326, upper bound: 3.3361256
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3511452, upper bound: 3.3526375
time: 0.41 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.9057254, 1.5460010, -1.3967609, 2.6024876, -3.5082123, 2.9427619
1: -1.4441154, 2.2252870, -2.1924410, 3.6463518, -5.0904665, 4.4177279
2: -0.9860084, 2.2430625, -1.5335877, 3.7314696, -4.7174778, 3.7766502
3: -2.4779506, 2.8746738, -3.8901188, 4.6021409, -7.0800896, 6.7647924
4: -1.5732667, 2.9222274, -2.4601402, 4.7640634, -6.3373299, 5.3823676

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3846804, upper bound: 3.3522795
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3872651, upper bound: 3.3590411
time: 0.42 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.9148694, 1.5653143, -0.9845003, 1.7182462, -2.6331153, 2.5498147
1: -1.4585052, 2.2524974, -1.5601554, 2.4329228, -3.8914278, 3.8126528
2: -0.9964871, 2.2692239, -1.0574925, 2.4967258, -3.4932129, 3.3267164
3: -2.5034299, 2.9087937, -2.7271707, 3.1218429, -5.6252718, 5.6359639
4: -1.5905116, 2.9554746, -1.6859249, 3.2546177, -4.8451285, 4.6413994

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3480290, upper bound: 3.3438455
time: 0.39 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3506071, upper bound: 3.3506071
time: 0.42 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.04 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.04
Output dim: 0, lower bound: -3.3821391, upper bound: 3.3444861
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.04
Output dim: 0, lower bound: -3.3877086, upper bound: 3.3609976
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.04
Output dim: 0, lower bound: -3.3456326, upper bound: 3.3361256
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.04
Output dim: 0, lower bound: -3.3511452, upper bound: 3.3526375
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.04
Output dim: 0, lower bound: -3.3846804, upper bound: 3.3522795
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.04
Output dim: 0, lower bound: -3.3872651, upper bound: 3.3590411
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.04
Output dim: 0, lower bound: -3.3480290, upper bound: 3.3438455
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.04
Output dim: 0, lower bound: -3.3506071, upper bound: 3.3506071

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.5404496, 0.7971597, -1.3827021, 2.5748899, -3.1153395, 2.1798618
1: -0.8714017, 1.1110208, -2.1702476, 3.6076930, -4.4790945, 3.2812684
2: -0.5814779, 1.2113867, -1.5165050, 3.6925826, -4.2740598, 2.7278914
3: -1.4605370, 1.4797201, -3.8523226, 4.5537357, -6.0142722, 5.3320427
4: -0.8908472, 1.6229602, -2.4324727, 4.7163944, -5.6072407, 4.0554314

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_A1

### Relational analysis result of IS_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3684833, upper bound: 3.3069356
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A1_A2

### Relational analysis result of IS_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3738711, upper bound: 3.3287107
time: 0.40 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.7670872, 1.2498345, -1.4205828, 2.6549139, -3.4220009, 2.6704173
1: -1.2188511, 1.7433900, -2.2293470, 3.7171719, -4.9360218, 3.9727368
2: -0.8199438, 1.8607113, -1.5606055, 3.8029828, -4.6229267, 3.4213164
3: -2.1090906, 2.2712719, -3.9579871, 4.6879396, -6.7970304, 6.2292576
4: -1.2925965, 2.4423008, -2.5043542, 4.8521762, -6.1447725, 4.9466553

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 37

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3785725, upper bound: 3.3532758
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A2_A2

### Relational analysis result of IS_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3682808, upper bound: 3.3174748
time: 0.42 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.5554659, 0.8305931, -0.9777416, 1.7077253, -2.2631912, 1.8083347
1: -0.8957945, 1.1573206, -1.5497024, 2.4163616, -3.3121560, 2.7070227
2: -0.5975975, 1.2580862, -1.0489711, 2.4826696, -3.0802670, 2.3070567
3: -1.5058055, 1.5370578, -2.7120686, 3.0995097, -4.6053143, 4.2491250
4: -0.9185979, 1.6823144, -1.6718051, 3.2363486, -4.1549463, 3.3541195

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3096470, upper bound: 3.3124485
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3267827, upper bound: 3.3180728
time: 0.42 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.7844816, 1.2864925, -1.0157527, 1.7855062, -2.5699878, 2.3022451
1: -1.2457039, 1.7942019, -1.6081196, 2.5250821, -3.7707860, 3.4023213
2: -0.8395393, 1.9117866, -1.0928304, 2.5890822, -3.4286213, 3.0046170
3: -2.1576509, 2.3350203, -2.8151433, 3.2343123, -5.3919630, 5.1501627
4: -1.3245426, 2.5065725, -1.7422895, 3.3683147, -4.6928573, 4.2488623

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_A1

### Relational analysis result of IS_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3357296, upper bound: 3.3437098
time: 0.44 seconds

## Relational analysis of IS_A1_B2_A2_A2

### Relational analysis result of IS_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3274894, upper bound: 3.3080332
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.7594744, 1.2632136, -1.3590050, 2.5225174, -3.2819917, 2.6222184
1: -1.2302890, 1.7965086, -2.1335068, 3.5369425, -4.7672300, 3.9300153
2: -0.8244767, 1.8348620, -1.4896221, 3.6211526, -4.4456291, 3.3244832
3: -2.0822861, 2.3361163, -3.7846868, 4.4681549, -6.5504398, 6.1208024
4: -1.2966200, 2.4227548, -2.3884292, 4.6284704, -5.9250903, 4.8111830

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_A1

### Relational analysis result of IS_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3749693, upper bound: 3.3416501
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_A2

### Relational analysis result of IS_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3757598, upper bound: 3.3408539
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.8414593, 1.4099056, -1.3967609, 2.6024876, -3.4439461, 2.8066664
1: -1.3441256, 2.0345333, -2.1924410, 3.6463518, -4.9904761, 4.2269740
2: -0.9120713, 2.0554502, -1.5335877, 3.7314696, -4.6435409, 3.5890379
3: -2.2974017, 2.6352177, -3.8901188, 4.6021409, -6.8995419, 6.5253363
4: -1.4497966, 2.6854737, -2.4601402, 4.7640634, -6.2138591, 5.1456137

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3807682, upper bound: 3.3453681
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3357422, upper bound: 3.3353811
time: 0.40 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.7681915, 1.2819659, -0.9466221, 1.6437194, -2.4119108, 2.2285879
1: -1.2440100, 1.8226275, -1.5019202, 2.3272271, -3.5712368, 3.3245478
2: -0.8343886, 1.8598800, -1.0140865, 2.3935182, -3.2279067, 2.8739665
3: -2.1064792, 2.3688455, -2.6245070, 2.9881086, -5.0945878, 4.9933524
4: -1.3129199, 2.4546323, -1.6158726, 3.1237268, -4.4366465, 4.0705051

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A1_A1

### Relational analysis result of IS_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3470913, upper bound: 3.3432347
time: 0.43 seconds

## Relational analysis of IS_A2_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3120792, upper bound: 3.3167216
time: 0.39 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3288778, upper bound: 3.3222956
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.8505706, 1.4291950, -0.9845003, 1.7182462, -2.5688167, 2.4136953
1: -1.3584607, 2.0616763, -1.5601554, 2.4329228, -3.7913835, 3.6218317
2: -0.9225284, 2.0815630, -1.0574925, 2.4967258, -3.4192543, 3.1390555
3: -2.3227718, 2.6692722, -2.7271707, 3.1218429, -5.4446130, 5.3964429
4: -1.4669735, 2.7186108, -1.6859249, 3.2546177, -4.7215910, 4.4045358

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A2_A1

### Relational analysis result of IS_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3350871, upper bound: 3.3405015
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A2_A2

### Relational analysis result of IS_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3339016, upper bound: 3.3339016
time: 0.41 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.57 seconds
IS_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 3.57
Output dim: 0, lower bound: -3.3684833, upper bound: 3.3069356
IS_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 3.57
Output dim: 0, lower bound: -3.3738711, upper bound: 3.3287107
IS_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 3.57
Output dim: 0, lower bound: -3.3785725, upper bound: 3.3532758
IS_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 3.57
Output dim: 0, lower bound: -3.3682808, upper bound: 3.3174748
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.57
Output dim: 0, lower bound: -3.3096470, upper bound: 3.3124485
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.57
Output dim: 0, lower bound: -3.3267827, upper bound: 3.3180728
IS_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 3.57
Output dim: 0, lower bound: -3.3357296, upper bound: 3.3437098
IS_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 3.57
Output dim: 0, lower bound: -3.3274894, upper bound: 3.3080332
IS_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 3.57
Output dim: 0, lower bound: -3.3749693, upper bound: 3.3416501
IS_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 3.57
Output dim: 0, lower bound: -3.3757598, upper bound: 3.3408539
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.57
Output dim: 0, lower bound: -3.3807682, upper bound: 3.3453681
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.57
Output dim: 0, lower bound: -3.3357422, upper bound: 3.3353811
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.57
Output dim: 0, lower bound: -3.3120792, upper bound: 3.3167216
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.57
Output dim: 0, lower bound: -3.3288778, upper bound: 3.3222956
IS_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 3.57
Output dim: 0, lower bound: -3.3350871, upper bound: 3.3405015
IS_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 3.57
Output dim: 0, lower bound: -3.3339016, upper bound: 3.3339016

## BFS IS instance: IS_A1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.1566912, 0.1075858, -1.3827021, 2.5748899, -2.7315810, 1.4902879
1: -0.2073695, 0.1629432, -2.1702476, 3.6076930, -3.8150625, 2.3331907
2: -0.1880939, 0.1751805, -1.5165050, 3.6925826, -3.8806765, 1.6916856
3: -0.2733961, 0.1929052, -3.8523226, 4.5537357, -4.8271317, 4.0452271
4: -0.1752845, 0.2110046, -2.4324727, 4.7163944, -4.8916788, 2.6434767

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_A1_A1

### Relational analysis result of IS_A1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3655663, upper bound: 3.3061689
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_A1_B1

### Relational analysis result of IS_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3550424, upper bound: 3.2786002
time: 0.38 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2

### Relational analysis result of IS_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3509189, upper bound: 3.2781946
time: 0.48 seconds

## BFS IS instance: IS_A1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.3952934, 0.5675502, -1.3553820, 2.5183194, -2.9136128, 1.9229321
1: -0.6416427, 0.7865219, -2.1281137, 3.5310936, -4.1727352, 2.9146357
2: -0.4445042, 0.8641365, -1.4855410, 3.6140771, -4.0585814, 2.3496776
3: -1.0320108, 1.0388087, -3.7768295, 4.4598074, -5.4918184, 4.8156381
4: -0.6231043, 1.1451524, -2.3816280, 4.6194868, -5.2425904, 3.5267806

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_A2_B1

### Relational analysis result of IS_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3567346, upper bound: 3.2735555
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2

### Relational analysis result of IS_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3521910, upper bound: 3.2730408
time: 0.42 seconds

## BFS IS instance: IS_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.4184293, 0.5756162, -1.4205828, 2.6549139, -3.0733428, 1.9961988
1: -0.6790531, 0.8202943, -2.2293470, 3.7171719, -4.3962250, 3.0496411
2: -0.4584752, 0.8613749, -1.5606055, 3.8029828, -4.2614579, 2.4219804
3: -1.0855377, 1.1013691, -3.9579871, 4.6879396, -5.7734766, 5.0593557
4: -0.6808510, 1.1661190, -2.5043542, 4.8521762, -5.5330272, 3.6704731

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_A1_B1

### Relational analysis result of IS_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3679043, upper bound: 3.3351596
time: 0.45 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_A1_B1

### Relational analysis result of IS_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2865299, upper bound: 3.1059972
time: 0.44 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_A1_B1

### Relational analysis result of IS_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3579778, upper bound: 3.2765978
time: 0.45 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2

### Relational analysis result of IS_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3551089, upper bound: 3.2763470
time: 0.51 seconds

## BFS IS instance: IS_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.3640207, 0.4919163, -1.3785408, 2.5640163, -2.9280369, 1.8704571
1: -0.5919223, 0.6995754, -2.1639917, 3.5959010, -4.1878233, 2.8635671
2: -0.4064289, 0.7356837, -1.5123584, 3.6770403, -4.0834694, 2.2480421
3: -0.9236271, 0.9362826, -3.8384664, 4.5406985, -5.4643254, 4.7747488
4: -0.5841237, 0.9869423, -2.4261057, 4.6968131, -5.2809362, 3.4130478

Time for backsubstitution: 2.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_A2_A1

### Relational analysis result of IS_A1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3528656, upper bound: 3.2843889
time: 0.43 seconds

## Relational analysis of IS_A1_B1_A2_A2_A2

### Relational analysis result of IS_A1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3561462, upper bound: 3.2978423
time: 0.44 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.5554659, 0.8305931, -0.1959889, 0.2247578, -0.7802235, 1.0265820
1: -0.8957945, 1.1573206, -0.2806676, 0.3151467, -1.2109407, 1.4379883
2: -0.5975975, 1.2580862, -0.2408800, 0.3375724, -0.9351696, 1.4989662
3: -1.5058055, 1.5370578, -0.3852224, 0.3898511, -1.8956561, 1.9222802
4: -0.9185979, 1.6823144, -0.2412743, 0.4217824, -1.3403803, 1.9235888

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3049120, upper bound: 3.2912590
time: 0.39 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3049120, upper bound: 3.3124485
time: 0.37 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.5358094, 0.7888141, -0.8153586, 1.3967929, -1.9326020, 1.6041727
1: -0.8638341, 1.1002151, -1.3072987, 1.9597688, -2.8236029, 2.4075136
2: -0.5768042, 1.1996298, -0.8700274, 2.0611413, -2.6379447, 2.0696571
3: -1.4469945, 1.4655610, -2.3070908, 2.5122108, -3.9592044, 3.7726514
4: -0.8827759, 1.6065240, -1.3469694, 2.6976514, -3.5804272, 2.9534934

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 37

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3256297, upper bound: 3.3170350
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3213949, upper bound: 3.2962976
time: 0.47 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3213949, upper bound: 3.3180728
time: 0.45 seconds

## BFS IS instance: IS_A1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.4309523, 0.5937985, -1.0157527, 1.7855062, -2.2164583, 1.6095511
1: -0.6992674, 0.8453233, -1.6081196, 2.5250821, -3.2243495, 2.4534431
2: -0.4707711, 0.8920389, -1.0928304, 2.5890822, -3.0598531, 1.9848694
3: -1.1241608, 1.1358755, -2.8151433, 3.2343123, -4.3584723, 3.9510188
4: -0.7021792, 1.2080480, -1.7422895, 3.3683147, -4.0704937, 2.9503369

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_A1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3150576, upper bound: 3.2669264
time: 0.43 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2

### Relational analysis result of IS_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3129584, upper bound: 3.2661924
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.3713922, 0.5024018, -0.9784041, 1.7076430, -2.0790353, 1.4808059
1: -0.6046131, 0.7143217, -1.5507520, 2.4192348, -3.0238478, 2.2650735
2: -0.4140540, 0.7521989, -1.0504620, 2.4816241, -2.8956780, 1.8026609
3: -0.9467055, 0.9569057, -2.7114942, 3.1038826, -4.0505877, 3.6683998
4: -0.5968709, 1.0110101, -1.6751022, 3.2345295, -3.8313997, 2.6861119

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_A2_B1

### Relational analysis result of IS_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2768229, upper bound: 3.2770353
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2

### Relational analysis result of IS_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3047538, upper bound: 3.2854838
time: 0.44 seconds

## BFS IS instance: IS_A2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.4951113, 0.7032819, -1.3590050, 2.5225174, -3.0176287, 2.0622869
1: -0.8158489, 0.9950314, -2.1335068, 3.5369425, -4.3527899, 3.1285379
2: -0.5401000, 1.0682610, -1.4896221, 3.6211526, -4.1612525, 2.5578816
3: -1.3292471, 1.3387210, -3.7846868, 4.4681549, -5.7974019, 5.1234078
4: -0.8108875, 1.4399529, -2.3884292, 4.6284704, -5.4393578, 3.8283818

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A1_A1_B1

### Relational analysis result of IS_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3732498, upper bound: 3.3369264
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_A1_B2

### Relational analysis result of IS_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3703809, upper bound: 3.3366756
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.3515007, 0.4941630, -1.3174229, 2.4324195, -2.7839203, 1.8115857
1: -0.5871807, 0.7190046, -2.0689769, 3.4168086, -4.0039892, 2.7879815
2: -0.4011723, 0.7294978, -1.4419417, 3.4963827, -3.8975544, 2.1714396
3: -0.8900065, 0.9587156, -3.6664367, 4.3227634, -5.2127690, 4.6251521
4: -0.5711600, 0.9614463, -2.3112919, 4.4748287, -5.0459886, 3.2727380

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_A2_A1

### Relational analysis result of IS_A2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3564640, upper bound: 3.3002663
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A1_A2_A2

### Relational analysis result of IS_A2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3631952, upper bound: 3.3221266
time: 0.45 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.8414593, 1.4099056, -0.5341043, 0.7733896, -1.6148489, 1.9440093
1: -1.3441256, 2.0345333, -0.8628553, 1.1374351, -2.4815607, 2.8973885
2: -0.9120713, 2.0554502, -0.5710644, 1.1833301, -2.0954013, 2.6265142
3: -2.2974017, 2.6352177, -1.4329275, 1.5103263, -3.8077278, 4.0681453
4: -1.4497966, 2.6854737, -0.8879060, 1.5660951, -3.0158916, 3.5733795

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 37

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3679276, upper bound: 3.3352686
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A2_B1_B1

### Relational analysis result of IS_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3805968, upper bound: 3.3452128
time: 0.44 seconds

## Relational analysis of IS_A2_B1_A2_B1_B2

### Relational analysis result of IS_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3775640, upper bound: 3.3442680
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.8414593, 1.4099056, -1.0763187, 1.9005142, -2.7419732, 2.4862237
1: -1.3441256, 2.0345333, -1.7036260, 2.6801429, -4.0242686, 3.7381592
2: -0.9120713, 2.0554502, -1.1723857, 2.7692282, -3.6812994, 3.2278359
3: -2.2974017, 2.6352177, -2.9810340, 3.4387836, -5.7361851, 5.6162515
4: -1.4497966, 2.6854737, -1.8729589, 3.5736849, -5.0234809, 4.5584326

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 37

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A2_B2_B1

### Relational analysis result of IS_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3351615, upper bound: 3.3352348
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B2_B2

### Relational analysis result of IS_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3351615, upper bound: 3.3352908
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.7681915, 1.2819659, -0.1913223, 0.2115829, -0.9797743, 1.4732882
1: -1.2440100, 1.8226275, -0.2721274, 0.2974833, -1.5414932, 2.0947549
2: -0.8343886, 1.8598800, -0.2347853, 0.3189517, -1.1533401, 2.0946653
3: -2.1064792, 2.3688455, -0.3667556, 0.3651886, -2.4716678, 2.7356012
4: -1.3129199, 2.4546323, -0.2304727, 0.3975748, -1.7104946, 2.6851048

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 27

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3085220, upper bound: 3.3048482
time: 0.39 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3085220, upper bound: 3.3167216
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.7456418, 1.2368635, -0.7889489, 1.3382543, -2.0838959, 2.0258124
1: -1.2094193, 1.7600670, -1.2664330, 1.8774540, -3.0868733, 3.0265000
2: -0.8092976, 1.7982286, -0.8420833, 1.9808559, -2.7901535, 2.6403115
3: -2.0448163, 2.2895951, -2.2298481, 2.4111879, -4.4560041, 4.5194426
4: -1.2717750, 2.3753557, -1.3001425, 2.5965412, -3.8683155, 3.6754978

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3277276, upper bound: 3.3213655
time: 0.45 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B2_B1

### Relational analysis result of IS_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3190994, upper bound: 3.3117848
time: 0.45 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2

### Relational analysis result of IS_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3110595, upper bound: 3.3097681
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.5611360, 0.8241687, -0.9845003, 1.7182462, -2.2793820, 1.8086690
1: -0.9118259, 1.1939331, -1.5601554, 2.4329228, -3.3447485, 2.7540884
2: -0.5990710, 1.2517779, -1.0574925, 2.4967258, -3.0957968, 2.3092704
3: -1.5267982, 1.5806659, -2.7271707, 3.1218429, -4.6486406, 4.3078365
4: -0.9262891, 1.6600218, -1.6859249, 3.2546177, -4.1809068, 3.3459461

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B2_A2_A1_B1

### Relational analysis result of IS_A2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3323855, upper bound: 3.3266431
time: 0.40 seconds

## Relational analysis of IS_A2_B2_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_A1_B1

### Relational analysis result of IS_A2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2867708, upper bound: 3.3119254
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A2_A1_B2

### Relational analysis result of IS_A2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3140457, upper bound: 3.3200691
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.4275745, 0.5755042, -0.9473274, 1.6436913, -2.0712657, 1.5228314
1: -0.7032069, 0.8393673, -1.5030479, 2.3302386, -3.0334454, 2.3424153
2: -0.4658949, 0.8682605, -1.0156198, 2.3927500, -2.8586445, 1.8838803
3: -1.1127229, 1.1280763, -2.6239724, 2.9926834, -4.1054058, 3.7520487
4: -0.6861109, 1.1673785, -1.6193012, 3.1220326, -3.8081436, 2.7866797

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_A2_B1

### Relational analysis result of IS_A2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2846477, upper bound: 3.3036599
time: 0.40 seconds

## Relational analysis of IS_A2_B2_A2_A2_B2

### Relational analysis result of IS_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3120291, upper bound: 3.3120291
time: 0.42 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.08 seconds
IS_A1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -3.3550424, upper bound: 3.2786002
IS_A1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -3.3509189, upper bound: 3.2781946
IS_A1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -3.3567346, upper bound: 3.2735555
IS_A1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -3.3521910, upper bound: 3.2730408
IS_A1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -3.3579778, upper bound: 3.2765978
IS_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -3.3551089, upper bound: 3.2763470
IS_A1_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -3.3528656, upper bound: 3.2843889
IS_A1_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -3.3561462, upper bound: 3.2978423
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -3.3049120, upper bound: 3.2912590
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -3.3049120, upper bound: 3.3124485
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -3.3213949, upper bound: 3.2962976
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -3.3213949, upper bound: 3.3180728
IS_A1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -3.3150576, upper bound: 3.2669264
IS_A1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -3.3129584, upper bound: 3.2661924
IS_A1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -3.2768229, upper bound: 3.2770353
IS_A1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -3.3047538, upper bound: 3.2854838
IS_A2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -3.3732498, upper bound: 3.3369264
IS_A2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -3.3703809, upper bound: 3.3366756
IS_A2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -3.3564640, upper bound: 3.3002663
IS_A2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -3.3631952, upper bound: 3.3221266
IS_A2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -3.3805968, upper bound: 3.3452128
IS_A2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -3.3775640, upper bound: 3.3442680
IS_A2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -3.3351615, upper bound: 3.3352348
IS_A2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -3.3351615, upper bound: 3.3352908
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -3.3085220, upper bound: 3.3048482
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -3.3085220, upper bound: 3.3167216
IS_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -3.3190994, upper bound: 3.3117848
IS_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -3.3110595, upper bound: 3.3097681
IS_A2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -3.2867708, upper bound: 3.3119254
IS_A2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -3.3140457, upper bound: 3.3200691
IS_A2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -3.2846477, upper bound: 3.3036599
IS_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -3.3120291, upper bound: 3.3120291

## BFS IS instance: IS_A1_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1566912, 0.1075858, -0.9318540, 1.6239235, -1.7806147, 1.0394397
1: -0.2073695, 0.1629432, -1.4747305, 2.3211002, -2.5284698, 1.6376733
2: -0.1880939, 0.1751805, -1.0092857, 2.3786180, -2.5667119, 1.1844662
3: -0.2733961, 0.1929052, -2.5861642, 2.9845572, -3.2579532, 2.7790692
4: -0.1752845, 0.2110046, -1.6207094, 3.0691164, -3.2444007, 1.8317140

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3527168, upper bound: 3.2785009
time: 0.49 seconds

## Relational analysis of IS_A1_B1_A1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3507175, upper bound: 3.2778632
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3507175, upper bound: 3.2778632
time: 0.38 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1551955, 0.0964212, -0.7502697, 1.2744584, -1.4296539, 0.8466908
1: -0.2036660, 0.1489705, -1.2052940, 1.8264577, -2.0301237, 1.3542645
2: -0.1861034, 0.1574614, -0.8033344, 1.8689821, -2.0550854, 0.9607958
3: -0.2661860, 0.1764608, -2.1021802, 2.3525188, -2.6187048, 2.2786407
4: -0.1724937, 0.1894732, -1.2774166, 2.4424112, -2.6149049, 1.4668899

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3367940, upper bound: 3.2753439
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3493501, upper bound: 3.2741851
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2_B2

### Relational analysis result of IS_A1_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3481101, upper bound: 3.2741991
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.3952934, 0.5675502, -1.2855784, 2.3641348, -2.7594283, 1.8531286
1: -0.6416427, 0.7865219, -2.0208244, 3.3168085, -3.9584510, 2.8073463
2: -0.4445042, 0.8641365, -1.4080541, 3.4050186, -3.8495224, 2.2721906
3: -1.0320108, 1.0388087, -3.5789158, 4.2026277, -5.2346382, 4.6177244
4: -0.6231043, 1.1451524, -2.2519004, 4.3616533, -4.9847574, 3.3970528

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346543, upper bound: 3.2689212
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A1_A2_B1_B2

### Relational analysis result of IS_A1_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346543, upper bound: 3.2735555
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.3848404, 0.5524802, -1.7336001, 3.3099432, -3.6947832, 2.2860804
1: -0.6250212, 0.7656825, -2.7282431, 4.5893450, -5.2143660, 3.4939256
2: -0.4346131, 0.8398435, -1.9285048, 4.7499909, -5.1846037, 2.7683482
3: -1.0008879, 1.0102435, -4.8628249, 5.7708011, -6.7716889, 5.8730683
4: -0.6061080, 1.1112640, -3.1034563, 6.0183010, -6.6244082, 4.2147202

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_A2_B2_B1

### Relational analysis result of IS_A1_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3521910, upper bound: 3.2730408
time: 0.46 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2_B2

### Relational analysis result of IS_A1_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3521910, upper bound: 3.2730408
time: 0.42 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.4184293, 0.5756162, -1.3504028, 2.5005808, -2.9190099, 1.9260190
1: -0.6790531, 0.8202943, -2.1211638, 3.5031409, -4.1821938, 2.9414577
2: -0.4584752, 0.8613749, -1.4830772, 3.5934620, -4.0519371, 2.3444519
3: -1.0855377, 1.1013691, -3.7594786, 4.4305840, -5.5161209, 4.8608475
4: -0.6808510, 1.1661190, -2.3753440, 4.5932875, -5.2741385, 3.5414629

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_A1_B1_B1

### Relational analysis result of IS_A1_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2502071, upper bound: 3.0426760
time: 0.38 seconds

## Relational analysis of IS_A1_B1_A2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3421640, upper bound: 3.2496897
time: 0.38 seconds

## Relational analysis of IS_A1_B1_A2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_A1_B1_B1

### Relational analysis result of IS_A1_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3574714, upper bound: 3.2764499
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A2_A1_B1_B2

### Relational analysis result of IS_A1_B1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3574714, upper bound: 3.2765978
time: 0.38 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.4095451, 0.5629041, -1.8015128, 3.4516218, -3.8611670, 2.3644171
1: -0.6649934, 0.8024540, -2.8340156, 4.7826977, -5.4476910, 3.6364696
2: -0.4499488, 0.8406522, -2.0068791, 4.9465876, -5.3965364, 2.8475308
3: -1.0589274, 1.0768943, -5.0535374, 6.0100870, -7.0690131, 6.1304307
4: -0.6660436, 1.1371961, -3.2314849, 6.2627544, -6.9287972, 4.3686790

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_A1_B2_B1

### Relational analysis result of IS_A1_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2489787, upper bound: 3.0425900
time: 0.38 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3409241, upper bound: 3.2497037
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A2_A1_B2_B1

### Relational analysis result of IS_A1_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3551089, upper bound: 3.2763470
time: 0.45 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2_B2

### Relational analysis result of IS_A1_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3551089, upper bound: 3.2763470
time: 0.50 seconds

## BFS IS instance: IS_A1_B1_A2_A2_A1

### Backsubstitution after applying IS history:
0: -0.1531387, 0.0685641, -1.3785408, 2.5640163, -2.7171550, 1.4471048
1: -0.2106493, 0.1112575, -2.1639917, 3.5959010, -3.8065503, 2.2752492
2: -0.1966404, 0.0946901, -1.5123584, 3.6770403, -3.8736808, 1.6070485
3: -0.2039572, 0.1406836, -3.8384664, 4.5406985, -4.7446556, 3.9791493
4: -0.1239186, 0.1213875, -2.4261057, 4.6968131, -4.8207316, 2.5474932

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_A2_A1_A1

### Relational analysis result of IS_A1_B1_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3387407, upper bound: 3.2815381
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A2_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_A2_A1_A1

### Relational analysis result of IS_A1_B1_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3335140, upper bound: 3.2797503
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A2_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_A2_A1_B1

### Relational analysis result of IS_A1_B1_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3492241, upper bound: 3.2731358
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A2_A2_A1_B2

### Relational analysis result of IS_A1_B1_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3479841, upper bound: 3.2731498
time: 0.43 seconds

## BFS IS instance: IS_A1_B1_A2_A2_A2

### Backsubstitution after applying IS history:
0: -0.2546977, 0.3192620, -1.3511612, 2.5072293, -2.7619271, 1.6704232
1: -0.3793527, 0.4361927, -2.1217225, 3.5193193, -3.8986721, 2.5579152
2: -0.3125294, 0.4806571, -1.4813521, 3.5979772, -3.9105065, 1.9620092
3: -0.5520888, 0.5574062, -3.7624331, 4.4469733, -4.9990621, 4.3198395
4: -0.3368047, 0.6047351, -2.3754995, 4.5992885, -4.9360933, 2.9802346

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_A2_A2_B1

### Relational analysis result of IS_A1_B1_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3468303, upper bound: 3.2655428
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A2_A2_A2_B2

### Relational analysis result of IS_A1_B1_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3461261, upper bound: 3.2655568
time: 0.52 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1580950, 0.1136471, -0.1959889, 0.2247578, -0.3828527, 0.3096360
1: -0.2104357, 0.1706810, -0.2806676, 0.3151467, -0.5255824, 0.4513486
2: -0.1899752, 0.1846488, -0.2408800, 0.3375724, -0.5275476, 0.4255288
3: -0.2780852, 0.2016969, -0.3852224, 0.3898511, -0.6679363, 0.5869193
4: -0.1770808, 0.2230911, -0.2412743, 0.4217824, -0.5988632, 0.4643654

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B1_A1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3047399, upper bound: 3.2906730
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2894639, upper bound: 3.2622563
time: 0.48 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2715156, upper bound: 3.2576924
time: 0.40 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.4111778, 0.5905373, -0.1959889, 0.2247578, -0.6359354, 0.7865262
1: -0.6676551, 0.8171499, -0.2806676, 0.3151467, -0.9828015, 1.0978174
2: -0.4602341, 0.9009085, -0.2408800, 0.3375724, -0.7978063, 1.1417885
3: -1.0806211, 1.0811872, -0.3852224, 0.3898511, -1.4704722, 1.4664097
4: -0.6492290, 1.1976668, -0.2412743, 0.4217824, -1.0710113, 1.4389410

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B1_A2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3047399, upper bound: 3.3124485
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B1_A2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2762236, upper bound: 3.2973092
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2715156, upper bound: 3.2673304
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.1580950, 0.1136471, -0.8153586, 1.3967929, -1.5548879, 0.9290057
1: -0.2104357, 0.1706810, -1.3072987, 1.9597688, -2.1702044, 1.4779794
2: -0.1899752, 0.1846488, -0.8700274, 2.0611413, -2.2511165, 1.0546759
3: -0.2780852, 0.2016969, -2.3070908, 2.5122108, -2.7902961, 2.5087876
4: -0.1770808, 0.2230911, -1.3469694, 2.6976514, -2.8747323, 1.5700604

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B2_A1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3047399, upper bound: 3.2955856
time: 0.44 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2894639, upper bound: 3.2678527
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2715156, upper bound: 3.2658361
time: 0.40 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.4111778, 0.5905373, -0.8153586, 1.3967929, -1.8079708, 1.4058959
1: -0.6676551, 0.8171499, -1.3072987, 1.9597688, -2.6274238, 2.1244481
2: -0.4602341, 0.9009085, -0.8700274, 2.0611413, -2.5213749, 1.7709359
3: -1.0806211, 1.0811872, -2.3070908, 2.5122108, -3.5928321, 3.3882780
4: -0.6492290, 1.1976668, -1.3469694, 2.6976514, -3.3468795, 2.5446358

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2856671, upper bound: 3.2556987
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2693954, upper bound: 3.2517214
time: 0.40 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.4309523, 0.5937985, -0.9534845, 1.6557400, -2.0866921, 1.5472828
1: -0.6992674, 0.8453233, -1.5133902, 2.3386474, -3.0379148, 2.3587136
2: -0.4707711, 0.8920389, -1.0246214, 2.4111423, -2.8819132, 1.9166603
3: -1.1241608, 1.1358755, -2.6419942, 3.0046701, -4.1288304, 3.7778697
4: -0.7021792, 1.2080480, -1.6303217, 3.1435275, -3.8457067, 2.8383698

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_A1_B1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2303165, upper bound: 3.0384832
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3133448, upper bound: 3.2659724
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A2_A1_B1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2892238, upper bound: 3.2605710
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A2_A1_B1_B2

### Relational analysis result of IS_A1_B2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2892238, upper bound: 3.2669264
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.4219694, 0.5807086, -1.2424016, 2.2724411, -2.6944103, 1.8231101
1: -0.6849757, 0.8272867, -1.9613321, 3.1862855, -3.8712611, 2.7886188
2: -0.4621466, 0.8693283, -1.3670864, 3.2689738, -3.7311203, 2.2364147
3: -1.0964459, 1.1112255, -3.4362047, 4.0423489, -5.1387944, 4.5474300
4: -0.6872090, 1.1775570, -2.1866059, 4.1725302, -4.8597383, 3.3641629

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_A1_B2_B1

### Relational analysis result of IS_A1_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2282173, upper bound: 3.0377492
time: 0.45 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2867225, upper bound: 3.2363556
time: 0.45 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A2_A1_B2_B1

### Relational analysis result of IS_A1_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2715547, upper bound: 3.2553438
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2_B2

### Relational analysis result of IS_A1_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2715547, upper bound: 3.2661924
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.3713922, 0.5024018, -0.1982261, 0.2298327, -0.6012248, 0.7006279
1: -0.6046131, 0.7143217, -0.2845629, 0.3215878, -0.9262009, 0.9988846
2: -0.4140540, 0.7521989, -0.2437147, 0.3445794, -0.7586334, 0.9959136
3: -0.9467055, 0.9569057, -0.3931998, 0.3987310, -1.3454362, 1.3501055
4: -0.5968709, 1.0110101, -0.2464022, 0.4311828, -1.0280535, 1.2574122

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2734623, upper bound: 3.2638866
time: 0.39 seconds

## Relational analysis of IS_A1_B2_A2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2734623, upper bound: 3.2770353
time: 0.50 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.3579228, 0.4839070, -0.8073455, 1.3800504, -1.7379730, 1.2912524
1: -0.5813816, 0.6870750, -1.2953401, 1.9382148, -2.5195961, 1.9824150
2: -0.4004894, 0.7237752, -0.8617237, 2.0377443, -2.4382336, 1.5854987
3: -0.9054074, 0.9184032, -2.2848239, 2.4851947, -3.3906021, 3.2032270
4: -0.5730715, 0.9690464, -1.3338299, 2.6668828, -3.2399542, 2.3028760

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3007299, upper bound: 3.2720303
time: 0.43 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3007299, upper bound: 3.2854838
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.4951113, 0.7032819, -1.2883730, 2.3673453, -2.8624566, 1.9916549
1: -0.8158489, 0.9950314, -2.0247960, 3.3216720, -4.1375208, 3.0198269
2: -0.5401000, 1.0682610, -1.4115394, 3.4102263, -3.9503257, 2.4798002
3: -1.3292471, 1.3387210, -3.5846338, 4.2095098, -5.5387568, 4.9233551
4: -0.8108875, 1.4399529, -2.2578506, 4.3682194, -5.1791067, 3.6978035

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3538110, upper bound: 3.3258182
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3567553, upper bound: 3.3025395
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3598878, upper bound: 3.3139233
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.4825817, 0.6806896, -1.7398125, 3.3212461, -3.8038273, 2.4205022
1: -0.7955853, 0.9629217, -2.7381759, 4.6049156, -5.4005008, 3.7010975
2: -0.5269998, 1.0354216, -1.9358580, 4.7661786, -5.2931786, 2.9712796
3: -1.2911263, 1.2971737, -4.8799090, 5.7905784, -7.0817046, 6.1770816
4: -0.7883824, 1.3951092, -3.1151023, 6.0392156, -6.8275981, 4.5102110

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3555154, upper bound: 3.3025510
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3586478, upper bound: 3.3139373
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A1_A2_A1

### Backsubstitution after applying IS history:
0: -0.1459303, 0.0571106, -1.3174229, 2.4324195, -2.5783498, 1.3745335
1: -0.2002850, 0.0829934, -2.0689769, 3.4168086, -3.6170936, 2.1519701
2: -0.1876060, 0.0749526, -1.4419417, 3.4963827, -3.6839888, 1.5168943
3: -0.1900112, 0.1067415, -3.6664367, 4.3227634, -4.5127745, 3.7731781
4: -0.1158497, 0.0948396, -2.3112919, 4.4748287, -4.5906782, 2.4061315

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 41

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A1_A2_A1_A1

### Relational analysis result of IS_A2_B1_A1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3373810, upper bound: 3.2955254
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_A2_A1_A1

### Relational analysis result of IS_A2_B1_A1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3422477, upper bound: 3.2963774
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A1_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A1_A2_A1_B1

### Relational analysis result of IS_A2_B1_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3562614, upper bound: 3.3001987
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A1_A2_A1_B2

### Relational analysis result of IS_A2_B1_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3550215, upper bound: 3.3002127
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A1_A2_A2

### Backsubstitution after applying IS history:
0: -0.2550131, 0.3627208, -1.2901304, 2.3759933, -2.6310060, 1.6528511
1: -0.4101242, 0.5034513, -2.0269265, 3.3405333, -3.7506566, 2.5303779
2: -0.3151606, 0.5291191, -1.4110171, 3.4176524, -3.7328129, 1.9401362
3: -0.5967675, 0.6484731, -3.5906355, 4.2295189, -4.8262858, 4.2391086
4: -0.3675103, 0.6654559, -2.2608242, 4.3779531, -4.7454634, 2.9262800

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 4

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A1_A2_A2_B1

### Relational analysis result of IS_A2_B1_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3626816, upper bound: 3.3206406
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A1_A2_A2_B2

### Relational analysis result of IS_A2_B1_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3615206, upper bound: 3.3206546
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.8414593, 1.4099056, -0.4536525, 0.6204764, -1.4619355, 1.8635581
1: -1.3441256, 2.0345333, -0.7323157, 0.9176537, -2.2617793, 2.7668483
2: -0.9120713, 2.0554502, -0.4875960, 0.9424510, -1.8545223, 2.5430460
3: -2.2974017, 2.6352177, -1.1725744, 1.2290919, -3.5264935, 3.8077919
4: -1.4497966, 2.6854737, -0.7383353, 1.2568524, -2.7066488, 3.4238091

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 37

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3677955, upper bound: 3.3351172
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3805968, upper bound: 3.3452128
time: 0.46 seconds

## Relational analysis of IS_A2_B1_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3805968, upper bound: 3.3452128
time: 0.49 seconds

## BFS IS instance: IS_A2_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.8414593, 1.4099056, -0.7005196, 1.1252439, -1.9667032, 2.1104250
1: -1.3441256, 2.0345333, -1.1131053, 1.6748258, -3.0189514, 3.1476386
2: -0.9120713, 2.0554502, -0.7371348, 1.6797721, -2.5918436, 2.7925849
3: -2.2974017, 2.6352177, -1.9085096, 2.1684570, -4.4658585, 4.5437264
4: -1.4497966, 2.6854737, -1.1901008, 2.1887615, -3.6385579, 3.8755746

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3571657, upper bound: 3.3328520
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A2_B1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_B2_B1

### Relational analysis result of IS_A2_B1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3656050, upper bound: 3.3419514
time: 0.45 seconds

## Relational analysis of IS_A2_B1_A2_B1_B2_B2

### Relational analysis result of IS_A2_B1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3656050, upper bound: 3.3420835
time: 0.45 seconds

## BFS IS instance: IS_A2_B1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.8414593, 1.4099056, -0.8015509, 1.3159778, -2.1574368, 2.2114561
1: -1.3441256, 2.0345333, -1.2684731, 1.8455193, -3.1896448, 3.3030064
2: -0.9120713, 2.0554502, -0.8579434, 1.9613158, -2.8733871, 2.9133935
3: -2.2974017, 2.6352177, -2.2049398, 2.3989067, -4.6963081, 4.8401566
4: -1.4497966, 2.6854737, -1.3557611, 2.5563803, -4.0061760, 4.0412350

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 37

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A2_B2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3351615, upper bound: 3.3352348
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A2_B2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3351615, upper bound: 3.3352348
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.8414593, 1.4099056, -0.8923019, 1.5116062, -2.3530650, 2.3022070
1: -1.3441256, 2.0345333, -1.4192272, 2.1590569, -3.5031824, 3.4537601
2: -0.9120713, 2.0554502, -0.9692066, 2.2144084, -3.1264794, 3.0246568
3: -2.2974017, 2.6352177, -2.4599638, 2.7914243, -5.0888257, 5.0951810
4: -1.4497966, 2.6854737, -1.5400366, 2.8657587, -4.3155551, 4.2255101

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 37

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3351615, upper bound: 3.3352908
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A2_B2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3351615, upper bound: 3.3352908
time: 0.44 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1818291, 0.1974240, -0.1913223, 0.2115829, -0.3934120, 0.3887463
1: -0.2557079, 0.2733605, -0.2721274, 0.2974833, -0.5531912, 0.5454880
2: -0.2230910, 0.2995206, -0.2347853, 0.3189517, -0.5420427, 0.5343059
3: -0.3370610, 0.3333832, -0.3667556, 0.3651886, -0.7022496, 0.7001388
4: -0.2180030, 0.3698795, -0.2304727, 0.3975748, -0.6155779, 0.6003521

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_A1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3082734, upper bound: 3.3042299
time: 0.39 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B1_A1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2795541, upper bound: 3.2888378
time: 0.39 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2770607, upper bound: 3.2797641
time: 0.47 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.5504941, 0.8429953, -0.1913223, 0.2115829, -0.7620770, 1.0343176
1: -0.9022513, 1.1839948, -0.2721274, 0.2974833, -1.1997346, 1.4561222
2: -0.6024966, 1.2809678, -0.2347853, 0.3189517, -0.9214480, 1.5157531
3: -1.5254459, 1.5534573, -0.3667556, 0.3651886, -1.8906345, 1.9202129
4: -0.8949078, 1.7017179, -0.2304727, 0.3975748, -1.2924823, 1.9321905

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B1_A2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2795541, upper bound: 3.3026492
time: 0.39 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2770607, upper bound: 3.3016245
time: 0.40 seconds

## BFS IS instance: IS_A2_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.7456418, 1.2368635, -0.4482742, 0.6271120, -1.3727539, 1.6851377
1: -1.2094193, 1.7600670, -0.7310713, 0.8830467, -2.0924659, 2.4911382
2: -0.8092976, 1.7982286, -0.4927061, 0.9659904, -1.7752877, 2.2909343
3: -2.0448163, 2.2895951, -1.1922607, 1.1721040, -3.2169201, 3.4818556
4: -1.2717750, 2.3753557, -0.6983207, 1.2887197, -2.5604942, 3.0736763

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2950090, upper bound: 3.2898145
time: 0.44 seconds

## Relational analysis of IS_A2_B2_A1_B2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2950090, upper bound: 3.3116475
time: 0.46 seconds

## BFS IS instance: IS_A2_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.7148981, 1.1719989, -0.3987091, 0.5581298, -1.2730278, 1.5707078
1: -1.1611781, 1.6721761, -0.6510427, 0.7797254, -1.9409035, 2.3232181
2: -0.7743154, 1.7119781, -0.4458783, 0.8555620, -1.6298772, 2.1578562
3: -1.9595007, 2.1789954, -1.0466853, 1.0277915, -2.9872921, 3.2256806
4: -1.2158325, 2.2636447, -0.6102455, 1.1291546, -2.3449862, 2.8738902

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B2_A1_B2_B2_B1

### Relational analysis result of IS_A2_B2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3105242, upper bound: 3.3082836
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2_B2

### Relational analysis result of IS_A2_B2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3073190, upper bound: 3.3073066
time: 0.40 seconds

## BFS IS instance: IS_A2_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.5611360, 0.8241687, -0.1970788, 0.2284780, -0.7896139, 1.0212475
1: -0.9118259, 1.1939331, -0.2828882, 0.3194023, -1.2312282, 1.4768214
2: -0.5990710, 1.2517779, -0.2422924, 0.3426604, -0.9417314, 1.4940703
3: -1.5267982, 1.5806659, -0.3908061, 0.3959427, -1.9227409, 1.9714720
4: -0.9262891, 1.6600218, -0.2447367, 0.4287302, -1.3550193, 1.9047585

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2811818, upper bound: 3.2945661
time: 0.44 seconds

## Relational analysis of IS_A2_B2_A2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2811818, upper bound: 3.3119254
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.5418269, 0.7863555, -0.8147271, 1.3950331, -1.9368601, 1.6010826
1: -0.8816208, 1.1382453, -1.3070034, 1.9581642, -2.8397844, 2.4452484
2: -0.5794773, 1.1963515, -0.8693506, 2.0582793, -2.6377563, 2.0657022
3: -1.4696215, 1.5112154, -2.3054674, 2.5104678, -3.9800887, 3.8166828
4: -0.8912935, 1.5903405, -1.3462176, 2.6944835, -3.5857766, 2.9365580

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B2_A2_A1_B2_B1

### Relational analysis result of IS_A2_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3099582, upper bound: 3.3033877
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3064666, upper bound: 3.3017339
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3064666, upper bound: 3.3200690
time: 0.44 seconds

## BFS IS instance: IS_A2_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.4275745, 0.5755042, -0.1933789, 0.2165971, -0.6441717, 0.7688830
1: -0.7032069, 0.8393673, -0.2758374, 0.3037848, -1.0069915, 1.1152047
2: -0.4658949, 0.8682605, -0.2373501, 0.3258314, -0.7917262, 1.1056106
3: -1.1127229, 1.1280763, -0.3746391, 0.3736754, -1.4863983, 1.5027153
4: -0.6861109, 1.1673785, -0.2341818, 0.4068049, -1.0929157, 1.4015603

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_A2_B1_B1

### Relational analysis result of IS_A2_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2844517, upper bound: 3.3025860
time: 0.43 seconds

## Relational analysis of IS_A2_B2_A2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A2_A2_B1_B1

### Relational analysis result of IS_A2_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2720303, upper bound: 3.3007299
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A2_A2_B1_B2

### Relational analysis result of IS_A2_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2720303, upper bound: 3.3036600
time: 0.44 seconds

## BFS IS instance: IS_A2_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.4118728, 0.5552812, -0.7805865, 1.3208041, -1.7326765, 1.3358676
1: -0.6786848, 0.8089522, -1.2539070, 1.8549881, -2.5336728, 2.0628588
2: -0.4512680, 0.8350543, -0.8334561, 1.9566174, -2.4078853, 1.6685104
3: -1.0673809, 1.0858749, -2.2066276, 2.3830485, -3.4504285, 3.2925024
4: -0.6597454, 1.1195757, -1.2865043, 2.5647061, -3.2244515, 2.4060800

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A2_A2_B2_B1

### Relational analysis result of IS_A2_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2854838, upper bound: 3.3047539
time: 0.43 seconds

## Relational analysis of IS_A2_B2_A2_A2_B2_B2

### Relational analysis result of IS_A2_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2854838, upper bound: 3.3120291
time: 0.45 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 5.26 seconds
IS_A1_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -3.3507175, upper bound: 3.2778632
IS_A1_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -3.3507175, upper bound: 3.2778632
IS_A1_B1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -3.3493501, upper bound: 3.2741851
IS_A1_B1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -3.3481101, upper bound: 3.2741991
IS_A1_B1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -3.3346543, upper bound: 3.2689212
IS_A1_B1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -3.3346543, upper bound: 3.2735555
IS_A1_B1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -3.3521910, upper bound: 3.2730408
IS_A1_B1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -3.3521910, upper bound: 3.2730408
IS_A1_B1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -3.3574714, upper bound: 3.2764499
IS_A1_B1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -3.3574714, upper bound: 3.2765978
IS_A1_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -3.3551089, upper bound: 3.2763470
IS_A1_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -3.3551089, upper bound: 3.2763470
IS_A1_B1_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -3.3492241, upper bound: 3.2731358
IS_A1_B1_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -3.3479841, upper bound: 3.2731498
IS_A1_B1_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -3.3468303, upper bound: 3.2655428
IS_A1_B1_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -3.3461261, upper bound: 3.2655568
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -3.2894639, upper bound: 3.2622563
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -3.2715156, upper bound: 3.2576924
IS_A1_B2_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -3.2762236, upper bound: 3.2973092
IS_A1_B2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -3.2715156, upper bound: 3.2673304
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -3.2894639, upper bound: 3.2678527
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -3.2715156, upper bound: 3.2658361
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -3.2856671, upper bound: 3.2556987
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -3.2693954, upper bound: 3.2517214
IS_A1_B2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -3.2892238, upper bound: 3.2605710
IS_A1_B2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -3.2892238, upper bound: 3.2669264
IS_A1_B2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -3.2715547, upper bound: 3.2553438
IS_A1_B2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -3.2715547, upper bound: 3.2661924
IS_A1_B2_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -3.2734623, upper bound: 3.2638866
IS_A1_B2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -3.2734623, upper bound: 3.2770353
IS_A1_B2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -3.3007299, upper bound: 3.2720303
IS_A1_B2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -3.3007299, upper bound: 3.2854838
IS_A2_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -3.3567553, upper bound: 3.3025395
IS_A2_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -3.3598878, upper bound: 3.3139233
IS_A2_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -3.3555154, upper bound: 3.3025510
IS_A2_B1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -3.3586478, upper bound: 3.3139373
IS_A2_B1_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -3.3562614, upper bound: 3.3001987
IS_A2_B1_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -3.3550215, upper bound: 3.3002127
IS_A2_B1_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -3.3626816, upper bound: 3.3206406
IS_A2_B1_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -3.3615206, upper bound: 3.3206546
IS_A2_B1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -3.3805968, upper bound: 3.3452128
IS_A2_B1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -3.3805968, upper bound: 3.3452128
IS_A2_B1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -3.3656050, upper bound: 3.3419514
IS_A2_B1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -3.3656050, upper bound: 3.3420835
IS_A2_B1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -3.3351615, upper bound: 3.3352348
IS_A2_B1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -3.3351615, upper bound: 3.3352348
IS_A2_B1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -3.3351615, upper bound: 3.3352908
IS_A2_B1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -3.3351615, upper bound: 3.3352908
IS_A2_B2_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -3.2795541, upper bound: 3.2888378
IS_A2_B2_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -3.2770607, upper bound: 3.2797641
IS_A2_B2_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -3.2795541, upper bound: 3.3026492
IS_A2_B2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -3.2770607, upper bound: 3.3016245
IS_A2_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -3.2950090, upper bound: 3.2898145
IS_A2_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -3.2950090, upper bound: 3.3116475
IS_A2_B2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -3.3105242, upper bound: 3.3082836
IS_A2_B2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -3.3073190, upper bound: 3.3073066
IS_A2_B2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -3.2811818, upper bound: 3.2945661
IS_A2_B2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -3.2811818, upper bound: 3.3119254
IS_A2_B2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -3.3064666, upper bound: 3.3017339
IS_A2_B2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -3.3064666, upper bound: 3.3200690
IS_A2_B2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -3.2720303, upper bound: 3.3007299
IS_A2_B2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -3.2720303, upper bound: 3.3036600
IS_A2_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -3.2854838, upper bound: 3.3047539
IS_A2_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -3.2854838, upper bound: 3.3120291

## BFS IS instance: IS_A1_B1_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1413234, 0.0555155, -0.9318540, 1.6239235, -1.7652469, 0.9873694
1: -0.1928025, 0.0894839, -1.4747305, 2.3211002, -2.5139027, 1.5642143
2: -0.1805860, 0.0825799, -1.0092857, 2.3786180, -2.5592041, 1.0918657
3: -0.1868363, 0.1145529, -2.5861642, 2.9845572, -3.1713934, 2.7007170
4: -0.1125539, 0.1099532, -1.6207094, 3.0691164, -3.1816702, 1.7306626

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_A1_B1_A1_A1

### Relational analysis result of IS_A1_B1_A1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3527168, upper bound: 3.2785009
time: 0.43 seconds

## Relational analysis of IS_A1_B1_A1_A1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_A1_B1_A1_A1

### Relational analysis result of IS_A1_B1_A1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3508022, upper bound: 3.2775906
time: 0.44 seconds

## Relational analysis of IS_A1_B1_A1_A1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3532406, upper bound: 3.2745422
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346881, upper bound: 3.2742247
time: 0.46 seconds

## Relational analysis of IS_A1_B1_A1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346881, upper bound: 3.2742247
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0715436, 0.0403326, -0.9318540, 1.6239235, -1.6954672, 0.9721865
1: -0.0824638, 0.0675079, -1.4747305, 2.3211002, -2.4035640, 1.5422384
2: -0.0682098, 0.0566018, -1.0092857, 2.3786180, -2.4468277, 1.0658875
3: -0.0799749, 0.0789351, -2.5861642, 2.9845572, -3.0645320, 2.6650991
4: -0.0637616, 0.0761483, -1.6207094, 3.0691164, -3.1328778, 1.6968577

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346881, upper bound: 3.2742247
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346881, upper bound: 3.2742247
time: 0.45 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.1551955, 0.0964212, -0.6874077, 1.1393424, -1.2945379, 0.7838289
1: -0.2036660, 0.1489705, -1.1069210, 1.6290243, -1.8326902, 1.2558913
2: -0.1861034, 0.1574614, -0.7368672, 1.6833022, -1.8694055, 0.8943287
3: -0.2661860, 0.1764608, -1.9218862, 2.1101232, -2.3763092, 2.0983467
4: -0.1724937, 0.1894732, -1.1659673, 2.2040489, -2.3765426, 1.3554406

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_A1_B2_B1_A1

### Relational analysis result of IS_A1_B1_A1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3350566, upper bound: 3.2712664
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_A1_B2_B1_A1

### Relational analysis result of IS_A1_B1_A1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3490542, upper bound: 3.2738365
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2_B1_A2

### Relational analysis result of IS_A1_B1_A1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3490542, upper bound: 3.2738365
time: 0.50 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.1547359, 0.0921981, -0.8401176, 1.5001327, -1.6548686, 0.9323157
1: -0.2024925, 0.1433489, -1.3569412, 2.1470032, -2.3494956, 1.5002902
2: -0.1854598, 0.1507975, -0.9059135, 2.1780326, -2.3634925, 1.0567111
3: -0.2635915, 0.1697158, -2.3873882, 2.7440710, -3.0076625, 2.5571036
4: -0.1712775, 0.1814092, -1.4576710, 2.8158877, -2.9871652, 1.6390803

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_A1_B2_B2_B1

### Relational analysis result of IS_A1_B1_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3481101, upper bound: 3.2741991
time: 0.45 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2_B2_B2

### Relational analysis result of IS_A1_B1_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3481101, upper bound: 3.2741991
time: 0.47 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.3952934, 0.5675502, -0.8992640, 1.5269049, -1.9221983, 1.4668142
1: -0.6416427, 0.7865219, -1.4145977, 2.1452823, -2.7869248, 2.2011197
2: -0.4445042, 0.8641365, -0.9673361, 2.2531991, -2.6977034, 1.8314724
3: -1.0320108, 1.0388087, -2.4788091, 2.7681799, -3.8001907, 3.5176177
4: -0.6231043, 1.1451524, -1.5321072, 2.9258275, -3.5489318, 2.6772594

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_B1

### Relational analysis result of IS_A1_B1_A1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3266414, upper bound: 3.2607566
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_A1

### Relational analysis result of IS_A1_B1_A1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3341316, upper bound: 3.2681953
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_A1

### Relational analysis result of IS_A1_B1_A1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346543, upper bound: 3.2689209
time: 0.49 seconds

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_A1

### Relational analysis result of IS_A1_B1_A1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346543, upper bound: 3.2689212
time: 0.45 seconds

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_A2

### Relational analysis result of IS_A1_B1_A1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346543, upper bound: 3.2689212
time: 0.41 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.3952934, 0.5675502, -1.0348082, 1.8135217, -2.2088151, 1.6023583
1: -0.6416427, 0.7865219, -1.6338042, 2.5872495, -3.2288921, 2.4203262
2: -0.4445042, 0.8641365, -1.1289939, 2.6354365, -3.0799406, 1.9931301
3: -1.0320108, 1.0388087, -2.8604848, 3.3247848, -4.3567953, 3.8992934
4: -0.6231043, 1.1451524, -1.8011415, 3.4072764, -4.0303807, 2.9462938

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_A1

### Relational analysis result of IS_A1_B1_A1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3341316, upper bound: 3.2681953
time: 0.43 seconds

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_B1

### Relational analysis result of IS_A1_B1_A1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3266414, upper bound: 3.2653304
time: 0.43 seconds

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_A1

### Relational analysis result of IS_A1_B1_A1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346543, upper bound: 3.2733328
time: 0.43 seconds

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_A1

### Relational analysis result of IS_A1_B1_A1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346543, upper bound: 3.2735555
time: 0.45 seconds

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_A2

### Relational analysis result of IS_A1_B1_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346543, upper bound: 3.2735555
time: 0.46 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.3848404, 0.5524802, -1.5274173, 2.9022233, -3.2870636, 2.0798974
1: -0.6250212, 0.7656825, -2.4179902, 4.0371771, -4.6621981, 3.1836727
2: -0.4346131, 0.8398435, -1.7061007, 4.1599712, -4.5945840, 2.5459442
3: -1.0008879, 1.0102435, -4.2765813, 5.0896196, -6.0905075, 5.2868247
4: -0.6061080, 1.1112640, -2.7457948, 5.2792945, -5.8854022, 3.8570583

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_A1

### Relational analysis result of IS_A1_B1_A1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3521910, upper bound: 3.2730408
time: 0.50 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_A2

### Relational analysis result of IS_A1_B1_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3521910, upper bound: 3.2730408
time: 0.43 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.3848404, 0.5524802, -1.7073554, 3.2505050, -3.6353450, 2.2598355
1: -0.6250212, 0.7656825, -2.6867869, 4.5096631, -5.1346841, 3.4524693
2: -0.4346131, 0.8398435, -1.8981711, 4.6682653, -5.1028776, 2.7380147
3: -1.0008879, 1.0102435, -4.7856188, 5.6740627, -6.6749506, 5.7958622
4: -0.6061080, 1.1112640, -3.0530100, 5.9169803, -6.5230880, 4.1642737

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_A2_B2_B2_A1

### Relational analysis result of IS_A1_B1_A1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3521910, upper bound: 3.2730408
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2_B2_A2

### Relational analysis result of IS_A1_B1_A1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3521910, upper bound: 3.2730408
time: 0.45 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.4184293, 0.5756162, -0.9104017, 1.5771786, -1.9956077, 1.4860177
1: -0.6790531, 0.8202943, -1.4433120, 2.2517207, -2.9307733, 2.2636063
2: -0.4584752, 0.8613749, -0.9884373, 2.3145106, -2.7729855, 1.8498122
3: -1.0855377, 1.1013691, -2.5247355, 2.9015985, -3.9871354, 3.6261046
4: -0.6808510, 1.1661190, -1.5855869, 2.9865203, -3.6673706, 2.7517056

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_B1

### Relational analysis result of IS_A1_B1_A2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2475579, upper bound: 3.0421904
time: 0.45 seconds

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_A1

### Relational analysis result of IS_A1_B1_A2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3418884, upper bound: 3.2493410
time: 0.44 seconds

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_B1

### Relational analysis result of IS_A1_B1_A2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3574597, upper bound: 3.2763642
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_B2

### Relational analysis result of IS_A1_B1_A2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3574597, upper bound: 3.2763642
time: 0.45 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.4184293, 0.5756162, -0.7142118, 1.1984377, -1.6168669, 1.2898278
1: -0.6790531, 0.8202943, -1.1496081, 1.7140702, -2.3931231, 1.9699018
2: -0.4584752, 0.8613749, -0.7657655, 1.7643851, -2.2228603, 1.6271403
3: -1.0855377, 1.1013691, -2.0007603, 2.2144721, -3.3000097, 3.1021295
4: -0.6808510, 1.1661190, -1.2143968, 2.3064671, -2.9873178, 2.3805156

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_A1_B1_B2_A1

### Relational analysis result of IS_A1_B1_A2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3418884, upper bound: 3.2496897
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A2_A1_B1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_A1_B1_B2_B1

### Relational analysis result of IS_A1_B1_A2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2475579, upper bound: 3.0426760
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A2_A1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_A1_B1_B2_A1

### Relational analysis result of IS_A1_B1_A2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3460926, upper bound: 3.2747547
time: 0.45 seconds

## Relational analysis of IS_A1_B1_A2_A1_B1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_A1_B1_B2_A1

### Relational analysis result of IS_A1_B1_A2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3399241, upper bound: 3.2727570
time: 0.43 seconds

## Relational analysis of IS_A1_B1_A2_A1_B1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A2_A1_B1_B2_B1

### Relational analysis result of IS_A1_B1_A2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3574597, upper bound: 3.2765978
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A2_A1_B1_B2_B2

### Relational analysis result of IS_A1_B1_A2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3574597, upper bound: 3.2765978
time: 0.43 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.4095451, 0.5629041, -1.5560662, 2.9602537, -3.3697989, 2.1189704
1: -0.6649934, 0.8024540, -2.4622838, 4.1165972, -4.7815905, 3.2647378
2: -0.4499488, 0.8406522, -1.7381214, 4.2417507, -4.6916995, 2.5787730
3: -1.0589274, 1.0768943, -4.3564076, 5.1875057, -6.2464328, 5.4333014
4: -0.6660436, 1.1371961, -2.7980089, 5.3816228, -6.0476661, 3.9352028

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_B1

### Relational analysis result of IS_A1_B1_A2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2489787, upper bound: 3.0425900
time: 0.44 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_A1

### Relational analysis result of IS_A1_B1_A2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3409241, upper bound: 3.2497037
time: 0.43 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_A1

### Relational analysis result of IS_A1_B1_A2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3551089, upper bound: 3.2763470
time: 0.43 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_A2

### Relational analysis result of IS_A1_B1_A2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3551089, upper bound: 3.2763470
time: 0.41 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.4095451, 0.5629041, -1.7361624, 3.3105764, -3.7201211, 2.2990665
1: -0.6649934, 0.8024540, -2.7317224, 4.5912924, -5.2562857, 3.5341763
2: -0.4499488, 0.8406522, -1.9310199, 4.7516222, -5.2015710, 2.7716720
3: -1.0589274, 1.0768943, -4.8668809, 5.7745323, -6.8334594, 5.9437742
4: -0.6660436, 1.1371961, -3.1064777, 6.0211258, -6.6871691, 4.2436719

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_B1

### Relational analysis result of IS_A1_B1_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2489787, upper bound: 3.0425900
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_A1

### Relational analysis result of IS_A1_B1_A2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3409241, upper bound: 3.2497037
time: 0.44 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_A1

### Relational analysis result of IS_A1_B1_A2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3551089, upper bound: 3.2763470
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_A2

### Relational analysis result of IS_A1_B1_A2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3551089, upper bound: 3.2763470
time: 0.49 seconds

## BFS IS instance: IS_A1_B1_A2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.1531387, 0.0685641, -1.3072083, 2.4073803, -2.5605190, 1.3757724
1: -0.2106493, 0.1112575, -2.0540006, 3.3791230, -3.5897722, 2.1652582
2: -0.1966404, 0.0946901, -1.4335508, 3.4640789, -3.6607194, 1.5282409
3: -0.2039572, 0.1406836, -3.6362844, 4.2800069, -4.4839640, 3.7769666
4: -0.1239186, 0.1213875, -2.2943711, 4.4338112, -4.5577297, 2.4157586

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=3.5238394737243652
rel_dist={0: [-3.3982802470538505, 3.398280247053849]}

## Binary search (step 1) starts
Candidate diff: 0.1250000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3972077, upper bound: 3.3971833
time: 0.44 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3967501, upper bound: 3.3967501
time: 0.42 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.10 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.10
Output dim: 0, lower bound: -3.3972077, upper bound: 3.3971833
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.10
Output dim: 0, lower bound: -3.3967501, upper bound: 3.3967501

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -1.1802469, 2.1300411, -1.2466472, 2.2771921, -3.4574389, 3.3766885
1: -1.8617817, 2.9912033, -1.9637374, 3.1950235, -5.0568037, 4.9549408
2: -1.2867744, 3.0743372, -1.3604455, 3.2710245, -4.5577965, 4.4347830
3: -3.2712717, 3.8114285, -3.4595599, 4.0564485, -7.3277192, 7.2709880
4: -2.0556865, 3.9620397, -2.1785955, 4.2066536, -6.2623401, 6.1406355

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9733663, upper bound: 3.0737807
time: 0.36 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3965613, upper bound: 3.3966073
time: 0.44 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -1.6433699, 3.1101141, -1.1789882, 2.1260657, -3.7694352, 4.2891026
1: -2.5895941, 4.3057761, -1.8591762, 2.9894636, -5.5790563, 6.1649523
2: -1.8265626, 4.4649234, -1.2829545, 3.0721710, -4.8987336, 5.7478771
3: -4.5898609, 5.4298372, -3.2686851, 3.8092656, -8.3991261, 8.6985226
4: -2.9406917, 5.6675196, -2.0519495, 3.9597485, -6.9004402, 7.7194691

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3136226, upper bound: 3.3759035
time: 0.53 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3963866, upper bound: 3.3963867
time: 0.40 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.21 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.21
Output dim: 0, lower bound: -2.9733663, upper bound: 3.0737807
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.21
Output dim: 0, lower bound: -3.3965613, upper bound: 3.3966073
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.21
Output dim: 0, lower bound: -3.3136226, upper bound: 3.3759035
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.21
Output dim: 0, lower bound: -3.3963866, upper bound: 3.3963867

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -1.0019149, 1.7509383, -0.7334445, 1.1887814, -2.1906962, 2.4843826
1: -1.5877810, 2.4751878, -1.1775849, 1.7064089, -3.2941892, 3.6527727
2: -1.0861900, 2.5480800, -0.7999445, 1.7725501, -2.8587401, 3.3480244
3: -2.7643802, 3.1825786, -2.0005958, 2.2385798, -5.0029597, 5.1831737
4: -1.7351170, 3.3051867, -1.2872248, 2.3013163, -4.0364327, 4.5924101

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9579689, upper bound: 3.0572657
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9368315, upper bound: 3.0078968
time: 0.39 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -1.1802469, 2.1300411, -1.1798793, 2.1306067, -3.3108535, 3.3099203
1: -1.8617817, 2.9912033, -1.8597046, 3.0020082, -4.8637900, 4.8509078
2: -1.2867744, 3.0743372, -1.2859197, 3.0733142, -4.3600864, 4.3602571
3: -3.2712717, 3.8114285, -3.2670975, 3.8226118, -7.0938835, 7.0785251
4: -2.0556865, 3.9620397, -2.0582790, 3.9565403, -6.0122271, 6.0203190

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3131461, upper bound: 3.3746531
time: 0.63 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3957785, upper bound: 3.3951257
time: 0.41 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -1.6433699, 3.1101141, -0.7930791, 1.3055665, -2.9489357, 3.9031932
1: -2.5895941, 4.3057761, -1.2595947, 1.8167671, -4.4063606, 5.5653706
2: -1.8265626, 4.4649234, -0.8499845, 1.9413579, -3.7679205, 5.3149080
3: -4.5898609, 5.4298372, -2.1844511, 2.3645957, -6.9544563, 7.6142883
4: -2.9406917, 5.6675196, -1.3427306, 2.5423412, -5.4830332, 7.0102501

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3124316, upper bound: 3.3758090
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3014810, upper bound: 3.3287864
time: 0.43 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -1.5050261, 2.8115311, -0.8613451, 1.4544559, -2.9594820, 3.6728761
1: -2.3767493, 3.9010277, -1.3774004, 2.0949125, -4.4716616, 5.2784271
2: -1.6711731, 4.0526819, -0.9366136, 2.1170239, -3.7881970, 4.9892950
3: -4.1974597, 4.9364190, -2.3571432, 2.7116809, -6.9091406, 7.2935619
4: -2.6863718, 5.1567984, -1.4906989, 2.7624111, -5.4487829, 6.6474972

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3492937, upper bound: 3.3851906
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3385786, upper bound: 3.3385786
time: 0.42 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.26 seconds
IS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 3.26
Output dim: 0, lower bound: -2.9579689, upper bound: 3.0572657
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 3.26
Output dim: 0, lower bound: -2.9368315, upper bound: 3.0078968
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 3.26
Output dim: 0, lower bound: -3.3131461, upper bound: 3.3746531
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 3.26
Output dim: 0, lower bound: -3.3957785, upper bound: 3.3951257
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.26
Output dim: 0, lower bound: -3.3124316, upper bound: 3.3758090
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.26
Output dim: 0, lower bound: -3.3014810, upper bound: 3.3287864
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.26
Output dim: 0, lower bound: -3.3492937, upper bound: 3.3851906
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.26
Output dim: 0, lower bound: -3.3385786, upper bound: 3.3385786

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: -1.1802469, 2.1300411, -0.7934000, 1.3059156, -2.4861619, 2.9234412
1: -1.8617817, 2.9912033, -1.2584804, 1.8233461, -3.6851277, 4.2496839
2: -1.2867744, 3.0743372, -0.8515308, 1.9400653, -3.2268388, 3.9258680
3: -3.2712717, 3.8114285, -2.1815679, 2.3729606, -5.6442323, 5.9929943
4: -2.0556865, 3.9620397, -1.3470855, 2.5378199, -4.5935044, 5.3091249

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3120034, upper bound: 3.3745830
time: 0.43 seconds

## Relational analysis of IS_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3026862, upper bound: 3.3351251
time: 0.45 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: -1.0334021, 1.8148571, -0.8685787, 1.4680140, -2.5014162, 2.6834359
1: -1.6348242, 2.5646641, -1.3866320, 2.1165228, -3.7513468, 3.9512961
2: -1.1204419, 2.6330025, -0.9448614, 2.1384559, -3.2588975, 3.5778635
3: -2.8490262, 3.2921867, -2.3752453, 2.7393909, -5.5884161, 5.6674318
4: -1.7903301, 3.4182785, -1.5064535, 2.7871442, -4.5774741, 4.9247322

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3489426, upper bound: 3.3839280
time: 0.63 seconds

## Relational analysis of IS_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3398237, upper bound: 3.3453040
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -1.8015128, 3.4516218, -0.7210956, 1.1545305, -2.9560432, 4.1727171
1: -2.8340156, 4.7826977, -1.1485593, 1.6070139, -4.4410286, 5.9312572
2: -2.0068791, 4.9465876, -0.7687967, 1.7308969, -3.7377760, 5.7153835
3: -5.0535374, 6.0100870, -1.9844897, 2.1013217, -7.1548586, 7.9945755
4: -3.2314849, 6.2627544, -1.2105491, 2.2779446, -5.5094290, 7.4733033

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3108362, upper bound: 3.3685955
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3085810, upper bound: 3.3685253
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -1.4036145, 2.6061134, -0.7930791, 1.3055665, -2.7091806, 3.3991926
1: -2.2199492, 3.6193051, -1.2595947, 1.8167671, -4.0367150, 4.8788991
2: -1.5485729, 3.7617416, -0.8499845, 1.9413579, -3.4899306, 4.6117263
3: -3.9271953, 4.5796375, -2.1844511, 2.3645957, -6.2917900, 6.7640886
4: -2.4795742, 4.8039675, -1.3427306, 2.5423412, -5.0219154, 6.1466975

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A2_A1

### Relational analysis result of IS_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2919242, upper bound: 3.2919242
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2_A2

### Relational analysis result of IS_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2919242, upper bound: 3.3287864
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -1.6914604, 3.2139149, -0.8244883, 1.3764974, -3.0679579, 4.0384030
1: -2.6636841, 4.4607854, -1.3194873, 1.9851868, -4.6488709, 5.7802725
2: -1.8827335, 4.6195955, -0.8943768, 2.0113480, -3.8940816, 5.5139723
3: -4.7410941, 5.6166272, -2.2545440, 2.5740318, -7.3151259, 7.8711705
4: -3.0278347, 5.8573508, -1.4213600, 2.6282899, -5.6561246, 7.2787099

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3477747, upper bound: 3.3783250
time: 0.39 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3476966, upper bound: 3.3785540
time: 0.40 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -1.2482872, 2.2745779, -0.8613451, 1.4544559, -2.7027426, 3.1359229
1: -1.9804271, 3.1710701, -1.3774004, 2.0949125, -4.0753393, 4.5484700
2: -1.3730903, 3.2922447, -0.9366136, 2.1170239, -3.4901142, 4.2288580
3: -3.4797421, 4.0320220, -2.3571432, 2.7116809, -6.1914225, 6.3891649
4: -2.1952660, 4.2211304, -1.4906989, 2.7624111, -4.9576769, 5.7118292

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_A1

### Relational analysis result of IS_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3090261, upper bound: 3.2999913
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A2_A2

### Relational analysis result of IS_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3139206, upper bound: 3.3139206
time: 0.40 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.13 seconds
IS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.13
Output dim: 0, lower bound: -3.3120034, upper bound: 3.3745830
IS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.13
Output dim: 0, lower bound: -3.3026862, upper bound: 3.3351251
IS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.13
Output dim: 0, lower bound: -3.3489426, upper bound: 3.3839280
IS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.13
Output dim: 0, lower bound: -3.3398237, upper bound: 3.3453040
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.13
Output dim: 0, lower bound: -3.3108362, upper bound: 3.3685955
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.13
Output dim: 0, lower bound: -3.3085810, upper bound: 3.3685253
IS_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 3.13
Output dim: 0, lower bound: -3.2919242, upper bound: 3.2919242
IS_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 3.13
Output dim: 0, lower bound: -3.2919242, upper bound: 3.3287864
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.13
Output dim: 0, lower bound: -3.3477747, upper bound: 3.3783250
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.13
Output dim: 0, lower bound: -3.3476966, upper bound: 3.3785540
IS_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 3.13
Output dim: 0, lower bound: -3.3090261, upper bound: 3.2999913
IS_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 3.13
Output dim: 0, lower bound: -3.3139206, upper bound: 3.3139206

## BFS IS instance: IS_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -1.3464081, 2.4858255, -0.7233083, 1.1585920, -2.5049996, 3.2091336
1: -2.1153784, 3.4856236, -1.1506581, 1.6191949, -3.7345734, 4.6362820
2: -1.4787391, 3.5821118, -0.7724853, 1.7350936, -3.2138329, 4.3545966
3: -3.7490828, 4.4103899, -1.9868397, 2.1162391, -5.8653216, 6.3972297
4: -2.3650093, 4.5791373, -1.2180914, 2.2801368, -4.6451464, 5.7972283

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_B1_A1_B1

### Relational analysis result of IS_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3114919, upper bound: 3.3739934
time: 0.43 seconds

## Relational analysis of IS_A1_B2_B1_A1_B2

### Relational analysis result of IS_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3088031, upper bound: 3.3716335
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.9534845, 1.6557400, -0.7934000, 1.3059156, -2.2593997, 2.4491401
1: -1.5133902, 2.3386474, -1.2584804, 1.8233461, -3.3367362, 3.5971279
2: -1.0246214, 2.4111423, -0.8515308, 1.9400653, -2.9646866, 3.2626731
3: -2.6419942, 3.0046701, -2.1815679, 2.3729606, -5.0149541, 5.1862369
4: -1.6303217, 3.1435275, -1.3470855, 2.5378199, -4.1681399, 4.4906130

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_B1_A2_A1

### Relational analysis result of IS_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3003245, upper bound: 3.3279117
time: 0.44 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2

### Relational analysis result of IS_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3003245, upper bound: 3.3351251
time: 0.43 seconds

## BFS IS instance: IS_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -1.2449485, 2.2677255, -0.8337846, 1.3947805, -2.6397290, 3.1015100
1: -1.9574517, 3.1875942, -1.3320740, 2.0133343, -3.9707842, 4.5196681
2: -1.3629357, 3.2760994, -0.9049069, 2.0390608, -3.4019961, 4.1810060
3: -3.4586945, 4.0493488, -2.2786522, 2.6097310, -6.0684252, 6.3280010
4: -2.1790338, 4.2031641, -1.4408650, 2.6610398, -4.8400736, 5.6440287

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_B2_A1_B1

### Relational analysis result of IS_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3483087, upper bound: 3.3832399
time: 0.42 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2

### Relational analysis result of IS_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3480747, upper bound: 3.3817742
time: 0.42 seconds

## BFS IS instance: IS_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.8105565, 1.3573258, -0.8685787, 1.4680140, -2.2785704, 2.2259045
1: -1.2932212, 1.9239732, -1.3866320, 2.1165228, -3.4097438, 3.3106050
2: -0.8653427, 1.9971926, -0.9448614, 2.1384559, -3.0037982, 2.9420538
3: -2.2380986, 2.4883218, -2.3752453, 2.7393909, -4.9774890, 4.8635674
4: -1.3729676, 2.6228518, -1.5064535, 2.7871442, -4.1601119, 4.1293049

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B2_A2_A1

### Relational analysis result of IS_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3105371, upper bound: 3.3067102
time: 0.41 seconds

## Relational analysis of IS_A1_B2_B2_A2_A2

### Relational analysis result of IS_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3162831, upper bound: 3.3240670
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -1.6506783, 3.1370323, -0.4467592, 0.6283662, -2.2790446, 3.5837915
1: -2.5993731, 4.3526740, -0.7209899, 0.8762589, -3.4756320, 5.0736637
2: -1.8317239, 4.5088830, -0.4858180, 0.9518604, -2.7835841, 4.9947004
3: -4.6305547, 5.4769077, -1.1724333, 1.1794225, -5.8099771, 6.6493411
4: -2.9450130, 5.7203417, -0.7276473, 1.2854217, -4.2304349, 6.4479890

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B1_B1

### Relational analysis result of IS_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3012832, upper bound: 3.3619203
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A1_B1_B2

### Relational analysis result of IS_A2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3001476, upper bound: 3.3619820
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -1.8015128, 3.4516218, -0.6645309, 1.0342084, -2.8357208, 4.1161528
1: -2.8340156, 4.7826977, -1.0604863, 1.4402702, -4.2742858, 5.8431840
2: -2.0068791, 4.9465876, -0.7053783, 1.5615146, -3.5683937, 5.6519661
3: -5.0535374, 6.0100870, -1.8231164, 1.8920205, -6.9455571, 7.8332014
4: -3.2314849, 6.2627544, -1.1047482, 2.0656259, -5.2971087, 7.3675017

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3085810, upper bound: 3.3685253
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3085810, upper bound: 3.3685253
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -1.0248677, 1.7878647, -0.7930791, 1.3055665, -2.3304341, 2.5809438
1: -1.6283838, 2.4675908, -1.2595947, 1.8167671, -3.4451509, 3.7271845
2: -1.1197983, 2.6275902, -0.8499845, 1.9413579, -3.0611563, 3.4775748
3: -2.8452539, 3.1804914, -2.1844511, 2.3645957, -5.2098494, 5.3649411
4: -1.7788833, 3.3934889, -1.3427306, 2.5423412, -4.3212242, 4.7362185

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_A1_B1

### Relational analysis result of IS_A2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2804764, upper bound: 3.2791060
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A2_A1_B2

### Relational analysis result of IS_A2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2791677, upper bound: 3.2791677
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -1.1667678, 2.0968153, -0.7930791, 1.3055665, -2.4723341, 2.8898942
1: -1.8632492, 2.9540858, -1.2595947, 1.8167671, -3.6800163, 4.2136798
2: -1.2962127, 3.0318215, -0.8499845, 1.9413579, -3.2375705, 3.8818059
3: -3.2405753, 3.7822700, -2.1844511, 2.3645957, -5.6051707, 5.9667201
4: -2.0667837, 3.8918052, -1.3427306, 2.5423412, -4.6091242, 5.2345352

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_A2_B1

### Relational analysis result of IS_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2804764, upper bound: 3.3197698
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2_A2_B2

### Relational analysis result of IS_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2791677, upper bound: 3.3198315
time: 0.47 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -1.5499620, 2.9211068, -0.6755250, 1.0847210, -2.6346827, 3.5966313
1: -2.4442379, 4.0590744, -1.0995250, 1.5506511, -3.9948890, 5.1585994
2: -1.7187898, 4.2115712, -0.7311220, 1.5951567, -3.3139462, 4.9426928
3: -4.3464975, 5.1185570, -1.8472068, 2.0276408, -6.3741384, 6.9657640
4: -2.7597313, 5.3520274, -1.1426152, 2.1133008, -4.8730321, 6.4946418

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B1_B1

### Relational analysis result of IS_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3228066, upper bound: 3.3672342
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A1_B1_B2

### Relational analysis result of IS_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3404451, upper bound: 3.3727183
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -1.6914604, 3.2139149, -0.7588080, 1.2385157, -2.9299760, 3.9727230
1: -2.6636841, 4.4607854, -1.2176425, 1.7909944, -4.4546785, 5.6784277
2: -1.8827335, 4.6195955, -0.8189517, 1.8199275, -3.7026610, 5.4385471
3: -4.7410941, 5.6166272, -2.0721469, 2.3291507, -7.0702448, 7.6887736
4: -3.0278347, 5.8573508, -1.2961447, 2.3868814, -5.4147148, 7.1534953

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3476966, upper bound: 3.3785540
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3476966, upper bound: 3.3785540
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.3146368, 0.4458627, -0.8613451, 1.4544559, -1.7690926, 1.3072076
1: -0.5149410, 0.6282328, -1.3774004, 2.0949125, -2.6098533, 2.0056326
2: -0.3713157, 0.6508200, -0.9366136, 2.1170239, -2.4883394, 1.5874333
3: -0.7655007, 0.8296257, -2.3571432, 2.7116809, -3.4771814, 3.1867690
4: -0.5041180, 0.8498913, -1.4906989, 2.7624111, -3.2665288, 2.3405898

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_A1_B1

### Relational analysis result of IS_A2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2962761, upper bound: 3.2962761
time: 0.40 seconds

## Relational analysis of IS_A2_B2_A2_A1_B2

### Relational analysis result of IS_A2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2962761, upper bound: 3.2999913
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -1.0284628, 1.8564339, -0.7286762, 1.1890892, -2.2175519, 2.5851099
1: -1.6531861, 2.5671122, -1.1753591, 1.7151535, -3.3683395, 3.7424710
2: -1.1282125, 2.7121873, -0.7863138, 1.7496284, -2.8778400, 3.4985008
3: -2.9212427, 3.2793324, -2.0013092, 2.2321467, -5.1533890, 5.2806416
4: -1.7709750, 3.4839532, -1.2434354, 2.3000062, -4.0709810, 4.7273884

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_A2_B1

### Relational analysis result of IS_A2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2999912, upper bound: 3.3090261
time: 0.40 seconds

## Relational analysis of IS_A2_B2_A2_A2_B2

### Relational analysis result of IS_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2999912, upper bound: 3.3139206
time: 0.41 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.18 seconds
IS_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.18
Output dim: 0, lower bound: -3.3114919, upper bound: 3.3739934
IS_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.18
Output dim: 0, lower bound: -3.3088031, upper bound: 3.3716335
IS_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.18
Output dim: 0, lower bound: -3.3003245, upper bound: 3.3279117
IS_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.18
Output dim: 0, lower bound: -3.3003245, upper bound: 3.3351251
IS_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.18
Output dim: 0, lower bound: -3.3483087, upper bound: 3.3832399
IS_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.18
Output dim: 0, lower bound: -3.3480747, upper bound: 3.3817742
IS_A1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.18
Output dim: 0, lower bound: -3.3105371, upper bound: 3.3067102
IS_A1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.18
Output dim: 0, lower bound: -3.3162831, upper bound: 3.3240670
IS_A2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.18
Output dim: 0, lower bound: -3.3012832, upper bound: 3.3619203
IS_A2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.18
Output dim: 0, lower bound: -3.3001476, upper bound: 3.3619820
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.18
Output dim: 0, lower bound: -3.3085810, upper bound: 3.3685253
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.18
Output dim: 0, lower bound: -3.3085810, upper bound: 3.3685253
IS_A2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.18
Output dim: 0, lower bound: -3.2804764, upper bound: 3.2791060
IS_A2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.18
Output dim: 0, lower bound: -3.2791677, upper bound: 3.2791677
IS_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.18
Output dim: 0, lower bound: -3.2804764, upper bound: 3.3197698
IS_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.18
Output dim: 0, lower bound: -3.2791677, upper bound: 3.3198315
IS_A2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.18
Output dim: 0, lower bound: -3.3228066, upper bound: 3.3672342
IS_A2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.18
Output dim: 0, lower bound: -3.3404451, upper bound: 3.3727183
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.18
Output dim: 0, lower bound: -3.3476966, upper bound: 3.3785540
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.18
Output dim: 0, lower bound: -3.3476966, upper bound: 3.3785540
IS_A2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.18
Output dim: 0, lower bound: -3.2962761, upper bound: 3.2962761
IS_A2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.18
Output dim: 0, lower bound: -3.2962761, upper bound: 3.2999913
IS_A2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.18
Output dim: 0, lower bound: -3.2999912, upper bound: 3.3090261
IS_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.18
Output dim: 0, lower bound: -3.2999912, upper bound: 3.3139206

## BFS IS instance: IS_A1_B2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -1.1956351, 2.1726060, -0.4579380, 0.6456896, -1.8413247, 2.6305437
1: -1.8804154, 3.0526402, -0.7393734, 0.9007239, -2.7811394, 3.7920136
2: -1.3031621, 3.1433032, -0.4968208, 0.9825131, -2.2856750, 3.6401234
3: -3.3276639, 3.8803093, -1.2084824, 1.2120626, -4.5397263, 5.0887914
4: -2.0793781, 4.0400887, -0.7472116, 1.3253480, -3.4047260, 4.7873001

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2996303, upper bound: 3.3626767
time: 0.39 seconds

## Relational analysis of IS_A1_B2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2998266, upper bound: 3.3639899
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -1.3464081, 2.4858255, -0.6665639, 1.0385145, -2.3849220, 3.1523895
1: -2.1153784, 3.4856236, -1.0625554, 1.4525832, -3.5679617, 4.5481787
2: -1.4787391, 3.5821118, -0.7086006, 1.5659763, -3.0447154, 4.2907124
3: -3.7490828, 4.4103899, -1.8259783, 1.9067502, -5.6558332, 6.2363672
4: -2.3650093, 4.5791373, -1.1116657, 2.0682950, -4.4333043, 5.6908021

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_B1_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2760100, upper bound: 3.3563685
time: 0.59 seconds

## Relational analysis of IS_A1_B2_B1_A1_B2_B2

### Relational analysis result of IS_A1_B2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2991120, upper bound: 3.3632888
time: 0.42 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.6151510, 0.9320602, -0.7934000, 1.3059156, -1.9210662, 1.7254603
1: -0.9845173, 1.2849982, -1.2584804, 1.8233461, -2.8078635, 2.5434785
2: -0.6501878, 1.4160383, -0.8515308, 1.9400653, -2.5902526, 2.2675691
3: -1.6948109, 1.6946074, -2.1815679, 2.3729606, -4.0677714, 3.8761742
4: -1.0021706, 1.8873043, -1.3470855, 2.5378199, -3.5399897, 3.2343898

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_B1_A2_A1_A1

### Relational analysis result of IS_A1_B2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2923672, upper bound: 3.3068736
time: 0.41 seconds

## Relational analysis of IS_A1_B2_B1_A2_A1_A2

### Relational analysis result of IS_A1_B2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2977522, upper bound: 3.3271636
time: 0.44 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.6416510, 0.9915570, -0.7934000, 1.3059156, -1.9475666, 1.7849571
1: -1.0385040, 1.4355035, -1.2584804, 1.8233461, -2.8618500, 2.6939840
2: -0.6867346, 1.4775138, -0.8515308, 1.9400653, -2.6267993, 2.3290446
3: -1.7519407, 1.8813579, -2.1815679, 2.3729606, -4.1249013, 4.0629234
4: -1.0710213, 1.9569094, -1.3470855, 2.5378199, -3.6088402, 3.3039949

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B1_A2_A2_B1

### Relational analysis result of IS_A1_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2714107, upper bound: 3.3113707
time: 0.41 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2_B2

### Relational analysis result of IS_A1_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2589131, upper bound: 3.3086897
time: 0.43 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -1.0974813, 1.9598112, -0.7046759, 1.1455956, -2.2430768, 2.6644871
1: -1.7284176, 2.7616868, -1.1443603, 1.6325371, -3.3609548, 3.9060471
2: -1.1917427, 2.8475709, -0.7626758, 1.6791778, -2.8709204, 3.6102464
3: -3.0469224, 3.5289857, -1.9313436, 2.1305788, -5.1775012, 5.4603291
4: -1.9017955, 3.6752985, -1.1952184, 2.2234364, -4.1252317, 4.8705168

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_B2_A1_B1_B1

### Relational analysis result of IS_A1_B2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3360652, upper bound: 3.3724874
time: 0.41 seconds

## Relational analysis of IS_A1_B2_B2_A1_B1_B2

### Relational analysis result of IS_A1_B2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3402501, upper bound: 3.3741169
time: 0.40 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -1.2449485, 2.2677255, -0.7697797, 1.2606049, -2.5055532, 3.0375051
1: -1.9574517, 3.1875942, -1.2328054, 1.8236289, -3.7810793, 4.4203997
2: -1.3629357, 3.2760994, -0.8312731, 1.8523769, -3.2153118, 4.1073723
3: -3.4586945, 4.0493488, -2.1008234, 2.3705170, -5.8292112, 6.1501722
4: -2.1790338, 4.2031641, -1.3185239, 2.4256351, -4.6046677, 5.5216880

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3479312, upper bound: 3.3817742
time: 0.39 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3479312, upper bound: 3.3817742
time: 0.43 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.1723508, 0.1510008, -0.8685787, 1.4680140, -1.6403648, 1.0195794
1: -0.2368947, 0.2176939, -1.3866320, 2.1165228, -2.3534174, 1.6043258
2: -0.2098696, 0.2328565, -0.9448614, 2.1384559, -2.3483255, 1.1777178
3: -0.3114822, 0.2593150, -2.3752453, 2.7393909, -3.0508730, 2.6345603
4: -0.1928091, 0.2867447, -1.5064535, 2.7871442, -2.9799533, 1.7931982

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B2_A2_A1_B1

### Relational analysis result of IS_A1_B2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2970341, upper bound: 3.3022958
time: 0.39 seconds

## Relational analysis of IS_A1_B2_B2_A2_A1_B2

### Relational analysis result of IS_A1_B2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2970341, upper bound: 3.3067102
time: 0.42 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.6521097, 1.0343874, -0.7470684, 1.2257402, -1.8778498, 1.7814554
1: -1.0542829, 1.4470388, -1.2017570, 1.7669888, -2.8212717, 2.6487956
2: -0.6988323, 1.5644706, -0.8067228, 1.8014688, -2.5003009, 2.3711929
3: -1.8297936, 1.8823802, -2.0509758, 2.2970846, -4.1268783, 3.9333546
4: -1.0567658, 2.0690625, -1.2776961, 2.3653893, -3.4221549, 3.3467586

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B2_A2_A2_B1

### Relational analysis result of IS_A1_B2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3013205, upper bound: 3.3182231
time: 0.39 seconds

## Relational analysis of IS_A1_B2_B2_A2_A2_B2

### Relational analysis result of IS_A1_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3013205, upper bound: 3.3240670
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -1.6506783, 3.1370323, -0.2476766, 0.3399331, -1.9906113, 3.3847089
1: -2.5993731, 4.3526740, -0.3936275, 0.4795916, -3.0789647, 4.7463017
2: -1.8317239, 4.5088830, -0.2906987, 0.5062616, -2.3379853, 4.7995811
3: -4.6305547, 5.4769077, -0.5775070, 0.6267461, -5.2573004, 6.0544147
4: -2.9450130, 5.7203417, -0.3863618, 0.6550677, -3.6000807, 6.1067033

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B1_B1_B1

### Relational analysis result of IS_A2_B1_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2753917, upper bound: 3.3482456
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A1_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A1_B1_B1_B1

### Relational analysis result of IS_A2_B1_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3012832, upper bound: 3.3619203
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A1_B1_B1_B2

### Relational analysis result of IS_A2_B1_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3012832, upper bound: 3.3619203
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -1.4304782, 2.6815753, -0.2202630, 0.2965953, -1.7270731, 2.9018383
1: -2.2607698, 3.7439203, -0.3436813, 0.4304437, -2.6912136, 4.0875998
2: -1.5827039, 3.8662219, -0.2650681, 0.4347911, -2.0174949, 4.1312895
3: -4.0128703, 4.7290335, -0.4853153, 0.5543633, -4.5672326, 5.2143478
4: -2.5427520, 4.9211960, -0.3405861, 0.5510295, -3.0937815, 5.2617822

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B1_B2_B1

### Relational analysis result of IS_A2_B1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2741991, upper bound: 3.3481102
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A1_B1_B2_B2

### Relational analysis result of IS_A2_B1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2655492, upper bound: 3.3451613
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -1.5560662, 2.9602537, -0.6645309, 1.0342084, -2.5902741, 3.6247845
1: -2.4622838, 4.1165972, -1.0604863, 1.4402702, -3.9025526, 5.1770830
2: -1.7381214, 4.2417507, -0.7053783, 1.5615146, -3.2996359, 4.9471292
3: -4.3564076, 5.1875057, -1.8231164, 1.8920205, -6.2484279, 7.0106220
4: -2.7980089, 5.3816228, -1.1047482, 2.0656259, -4.8636332, 6.4863710

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2763470, upper bound: 3.3551089
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2994077, upper bound: 3.3619588
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -1.7361624, 3.3105764, -0.6645309, 1.0342084, -2.7703707, 3.9751072
1: -2.7317224, 4.5912924, -1.0604863, 1.4402702, -4.1719923, 5.6517787
2: -1.9310199, 4.7516222, -0.7053783, 1.5615146, -3.4925344, 5.4570007
3: -4.8668809, 5.7745323, -1.8231164, 1.8920205, -6.7589006, 7.5976486
4: -3.1064777, 6.0211258, -1.1047482, 2.0656259, -5.1721020, 7.1258740

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2763470, upper bound: 3.3551089
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2994077, upper bound: 3.3619588
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -1.0248677, 1.7878647, -0.4331041, 0.5982436, -1.6231112, 2.2209687
1: -1.6283838, 2.4675908, -0.7036965, 0.8493544, -2.4777379, 3.1712868
2: -1.1197983, 2.6275902, -0.4736690, 0.9001736, -2.0199718, 3.1012592
3: -2.8452539, 3.1804914, -1.1340095, 1.1420630, -3.9873171, 4.3145008
4: -1.7788833, 3.3934889, -0.7076730, 1.2180282, -2.9969113, 4.1011620

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 37

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2553670, upper bound: 3.2722946
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2553438, upper bound: 3.2715547
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.8309133, 1.3851022, -0.3818156, 0.5175719, -1.3484848, 1.7669176
1: -1.3305371, 1.9255730, -0.6215257, 0.7316089, -2.0621459, 2.5470986
2: -0.8999594, 2.0549111, -0.4241971, 0.7762507, -1.6762102, 2.4791083
3: -2.3008118, 2.5067830, -0.9805481, 0.9820492, -3.2828610, 3.4873312
4: -1.4230094, 2.6764612, -0.6143144, 1.0468788, -2.4698882, 3.2907758

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2382175, upper bound: 3.2435948
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2365779, upper bound: 3.2365779
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -1.1667678, 2.0968153, -0.4331041, 0.5982436, -1.7650114, 2.5299194
1: -1.8632492, 2.9540858, -0.7036965, 0.8493544, -2.7126036, 3.6577814
2: -1.2962127, 3.0318215, -0.4736690, 0.9001736, -2.1963861, 3.5054905
3: -3.2405753, 3.7822700, -1.1340095, 1.1420630, -4.3826385, 4.9162793
4: -2.0667837, 3.8918052, -0.7076730, 1.2180282, -3.2848113, 4.5994782

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_A2_B1_B1

### Relational analysis result of IS_A2_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2613829, upper bound: 3.2931419
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2873719, upper bound: 3.3158699
time: 0.44 seconds

## Relational analysis of IS_A2_B1_A2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2873719, upper bound: 3.3197265
time: 0.50 seconds

## BFS IS instance: IS_A2_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.9673622, 1.6799760, -0.3818156, 0.5175719, -1.4849336, 2.0617912
1: -1.5564315, 2.3958812, -0.6215257, 0.7316089, -2.2880404, 3.0174067
2: -1.0676546, 2.4391298, -0.4241971, 0.7762507, -1.8439053, 2.8633270
3: -2.6740503, 3.0920553, -0.9805481, 0.9820492, -3.6560988, 4.0726032
4: -1.6978514, 3.1502225, -0.6143144, 1.0468788, -2.7447302, 3.7645364

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2422171, upper bound: 3.2585734
time: 0.44 seconds

## Relational analysis of IS_A2_B1_A2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2522087, upper bound: 3.2919245
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -1.5499620, 2.9211068, -0.4196987, 0.5819330, -2.1318951, 3.3408055
1: -2.4442379, 4.0590744, -0.6945332, 0.8248186, -3.2690563, 4.7536077
2: -1.7187898, 4.2115712, -0.4640239, 0.8772470, -2.5960369, 4.6755948
3: -4.3464975, 5.1185570, -1.1008496, 1.1121264, -5.4586239, 6.2194061
4: -2.7597313, 5.3520274, -0.6801515, 1.1812748, -3.9410062, 6.0321779

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B1_B1_B1

### Relational analysis result of IS_A2_B2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2965572, upper bound: 3.3540143
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A1_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B2_A1_B1_B1_B1

### Relational analysis result of IS_A2_B2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3228066, upper bound: 3.3672342
time: 0.43 seconds

## Relational analysis of IS_A2_B2_A1_B1_B1_B2

### Relational analysis result of IS_A2_B2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3228066, upper bound: 3.3672342
time: 0.44 seconds

## BFS IS instance: IS_A2_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -1.3322984, 2.4751472, -0.3037774, 0.4269910, -1.7592891, 2.7789247
1: -2.1097126, 3.4620111, -0.5060489, 0.6214350, -2.7311475, 3.9680600
2: -1.4727931, 3.5772123, -0.3532883, 0.6198184, -2.0926116, 3.9305005
3: -3.7357616, 4.3850961, -0.7409714, 0.8236294, -4.5593910, 5.1260676
4: -2.3652546, 4.5634842, -0.4891061, 0.8057628, -3.1710167, 5.0525904

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B1_B2_B1

### Relational analysis result of IS_A2_B2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2991403, upper bound: 3.3548499
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A1_B1_B2_B2

### Relational analysis result of IS_A2_B2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3206301, upper bound: 3.3614289
time: 0.46 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -1.4567093, 2.7474623, -0.7588080, 1.2385157, -2.6952248, 3.5062702
1: -2.3098607, 3.8269162, -1.2176425, 1.7909944, -4.1008549, 5.0445585
2: -1.6270657, 3.9458733, -0.8189517, 1.8199275, -3.4469929, 4.7648244
3: -4.0754924, 4.8336587, -2.0721469, 2.3291507, -6.4046426, 6.9058046
4: -2.6146660, 5.0151505, -1.2961447, 2.3868814, -5.0015473, 6.3112950

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3476916, upper bound: 3.3785540
time: 0.40 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3476916, upper bound: 3.3785540
time: 0.59 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -1.6257092, 3.0719349, -0.7588080, 1.2385157, -2.8642249, 3.8307428
1: -2.5608692, 4.2679744, -1.2176425, 1.7909944, -4.3518639, 5.4856167
2: -1.8065152, 4.4230766, -0.8189517, 1.8199275, -3.6264424, 5.2420282
3: -4.5531921, 5.3799329, -2.0721469, 2.3291507, -6.8823419, 7.4520798
4: -2.9023273, 5.6139865, -1.2961447, 2.3868814, -5.2892084, 6.9101310

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3476916, upper bound: 3.3783250
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3476916, upper bound: 3.3783250
time: 0.46 seconds

## BFS IS instance: IS_A2_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.3146368, 0.4458627, -0.1985916, 0.2254106, -0.5400473, 0.6444542
1: -0.5149410, 0.6282328, -0.2806932, 0.3091409, -0.8240817, 0.9089260
2: -0.3713157, 0.6508200, -0.2443213, 0.3333679, -0.7046834, 0.8951413
3: -0.7655007, 0.8296257, -0.3737890, 0.3830037, -1.1485044, 1.2034148
4: -0.5041180, 0.8498913, -0.2442167, 0.4127575, -0.9168755, 1.0941080

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A2_A1_B1_B1

### Relational analysis result of IS_A2_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2824701, upper bound: 3.2806231
time: 0.45 seconds

## Relational analysis of IS_A2_B2_A2_A1_B1_B2

### Relational analysis result of IS_A2_B2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2801292, upper bound: 3.2801292
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.3146368, 0.4458627, -0.6067135, 0.9378965, -1.2525332, 1.0525761
1: -0.5149410, 0.6282328, -0.9866807, 1.3224623, -1.8374027, 1.6149132
2: -0.3713157, 0.6508200, -0.6515975, 1.4241432, -1.7954589, 1.3024173
3: -0.7655007, 0.8296257, -1.7003349, 1.7263371, -2.4918375, 2.5299606
4: -0.5041180, 0.8498913, -0.9800219, 1.8836461, -2.3877637, 1.8299128

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A2_A1_B2_B1

### Relational analysis result of IS_A2_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2824701, upper bound: 3.2837556
time: 0.43 seconds

## Relational analysis of IS_A2_B2_A2_A1_B2_B2

### Relational analysis result of IS_A2_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2801292, upper bound: 3.2856642
time: 0.44 seconds

## BFS IS instance: IS_A2_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -1.0284628, 1.8564339, -0.1985916, 0.2254106, -1.2538732, 2.0550256
1: -1.6531861, 2.5671122, -0.2806932, 0.3091409, -1.9623270, 2.8478055
2: -1.1282125, 2.7121873, -0.2443213, 0.3333679, -1.4615804, 2.9565086
3: -2.9212427, 3.2793324, -0.3737890, 0.3830037, -3.3042462, 3.6531215
4: -1.7709750, 3.4839532, -0.2442167, 0.4127575, -2.1837325, 3.7281699

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2837555, upper bound: 3.2938538
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2856642, upper bound: 3.2991177
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -1.0284628, 1.8564339, -0.6067135, 0.9378965, -1.9663589, 2.4631474
1: -1.6531861, 2.5671122, -0.9866807, 1.3224623, -2.9756484, 3.5537925
2: -1.1282125, 2.7121873, -0.6515975, 1.4241432, -2.5523553, 3.3637841
3: -2.9212427, 3.2793324, -1.7003349, 1.7263371, -4.6475797, 4.9796672
4: -1.7709750, 3.4839532, -0.9800219, 1.8836461, -3.6546204, 4.4639750

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2448587, upper bound: 3.2511194
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2448587, upper bound: 3.3137138
time: 0.47 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.77 seconds
IS_A1_B2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -3.2996303, upper bound: 3.3626767
IS_A1_B2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -3.2998266, upper bound: 3.3639899
IS_A1_B2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -3.2760100, upper bound: 3.3563685
IS_A1_B2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -3.2991120, upper bound: 3.3632888
IS_A1_B2_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -3.2923672, upper bound: 3.3068736
IS_A1_B2_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -3.2977522, upper bound: 3.3271636
IS_A1_B2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -3.2714107, upper bound: 3.3113707
IS_A1_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -3.2589131, upper bound: 3.3086897
IS_A1_B2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -3.3360652, upper bound: 3.3724874
IS_A1_B2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -3.3402501, upper bound: 3.3741169
IS_A1_B2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -3.3479312, upper bound: 3.3817742
IS_A1_B2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -3.3479312, upper bound: 3.3817742
IS_A1_B2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -3.2970341, upper bound: 3.3022958
IS_A1_B2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -3.2970341, upper bound: 3.3067102
IS_A1_B2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -3.3013205, upper bound: 3.3182231
IS_A1_B2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -3.3013205, upper bound: 3.3240670
IS_A2_B1_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -3.3012832, upper bound: 3.3619203
IS_A2_B1_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -3.3012832, upper bound: 3.3619203
IS_A2_B1_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -3.2741991, upper bound: 3.3481102
IS_A2_B1_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -3.2655492, upper bound: 3.3451613
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -3.2763470, upper bound: 3.3551089
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -3.2994077, upper bound: 3.3619588
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -3.2763470, upper bound: 3.3551089
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -3.2994077, upper bound: 3.3619588
IS_A2_B1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -3.2553670, upper bound: 3.2722946
IS_A2_B1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -3.2553438, upper bound: 3.2715547
IS_A2_B1_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -3.2382175, upper bound: 3.2435948
IS_A2_B1_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -3.2365779, upper bound: 3.2365779
IS_A2_B1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -3.2873719, upper bound: 3.3158699
IS_A2_B1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -3.2873719, upper bound: 3.3197265
IS_A2_B1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -3.2422171, upper bound: 3.2585734
IS_A2_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -3.2522087, upper bound: 3.2919245
IS_A2_B2_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -3.3228066, upper bound: 3.3672342
IS_A2_B2_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -3.3228066, upper bound: 3.3672342
IS_A2_B2_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -3.2991403, upper bound: 3.3548499
IS_A2_B2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -3.3206301, upper bound: 3.3614289
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -3.3476916, upper bound: 3.3785540
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -3.3476916, upper bound: 3.3785540
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -3.3476916, upper bound: 3.3783250
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -3.3476916, upper bound: 3.3783250
IS_A2_B2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -3.2824701, upper bound: 3.2806231
IS_A2_B2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -3.2801292, upper bound: 3.2801292
IS_A2_B2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -3.2824701, upper bound: 3.2837556
IS_A2_B2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -3.2801292, upper bound: 3.2856642
IS_A2_B2_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -3.2837555, upper bound: 3.2938538
IS_A2_B2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -3.2856642, upper bound: 3.2991177
IS_A2_B2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -3.2448587, upper bound: 3.2511194
IS_A2_B2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -3.2448587, upper bound: 3.3137138

## BFS IS instance: IS_A1_B2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.7642612, 1.2836039, -0.4579380, 0.6456896, -1.4099509, 1.7415419
1: -1.2180088, 1.8319196, -0.7393734, 0.9007239, -2.1187327, 2.5712929
2: -0.8212187, 1.9048913, -0.4968208, 0.9825131, -1.8037317, 2.4017117
3: -2.1262181, 2.3707867, -1.2084824, 1.2120626, -3.3382807, 3.5792689
4: -1.3115227, 2.4698634, -0.7472116, 1.3253480, -2.6368709, 3.2170751

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2996012, upper bound: 3.3626767
time: 0.47 seconds

## Relational analysis of IS_A1_B2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2996012, upper bound: 3.3626767
time: 0.47 seconds

## BFS IS instance: IS_A1_B2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.6081191, 0.9629390, -0.3310670, 0.4620188, -1.0701379, 1.2940058
1: -0.9800584, 1.3768380, -0.5382158, 0.6617993, -1.6418577, 1.9150537
2: -0.6510510, 1.4463311, -0.3750983, 0.6836691, -1.3347199, 1.8214291
3: -1.6871425, 1.7991245, -0.8189269, 0.8794301, -2.5665724, 2.6180506
4: -1.0214297, 1.9009726, -0.5380028, 0.9042638, -1.9256935, 2.4389749

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2735803, upper bound: 3.3487259
time: 0.42 seconds

## Relational analysis of IS_A1_B2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2651580, upper bound: 3.3447562
time: 0.42 seconds

## BFS IS instance: IS_A1_B2_B1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -1.3464081, 2.4858255, -0.3499174, 0.4764199, -1.8228281, 2.8357427
1: -2.1153784, 3.4856236, -0.5662836, 0.6845342, -2.7999125, 4.0519071
2: -1.4787391, 3.5821118, -0.3910944, 0.7030903, -2.1818295, 3.9732058
3: -3.7490828, 4.4103899, -0.8726386, 0.9113393, -4.6604223, 5.2830286
4: -2.3650093, 4.5791373, -0.5642673, 0.9373554, -3.3023646, 5.1434045

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B1_A1_B2_B1_B1

### Relational analysis result of IS_A1_B2_B1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2491246, upper bound: 3.3413987
time: 0.41 seconds

## Relational analysis of IS_A1_B2_B1_A1_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_B1_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2754982, upper bound: 3.3532808
time: 0.43 seconds

## Relational analysis of IS_A1_B2_B1_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2754982, upper bound: 3.3563685
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_B1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -1.1454693, 2.0617070, -0.3227537, 0.4322443, -1.5777136, 2.3844607
1: -1.8037870, 2.9168172, -0.5191203, 0.6156684, -2.4194555, 3.4359367
2: -1.2485323, 2.9832628, -0.3634911, 0.6414436, -1.8899755, 3.3467534
3: -3.1765826, 3.7213202, -0.7908672, 0.8178735, -3.9944556, 4.5121870
4: -1.9976485, 3.8396754, -0.5117063, 0.8502003, -2.8478487, 4.3513808

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B1_A1_B2_B2_B1

### Relational analysis result of IS_A1_B2_B1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2726117, upper bound: 3.3485057
time: 0.41 seconds

## Relational analysis of IS_A1_B2_B1_A1_B2_B2_B2

### Relational analysis result of IS_A1_B2_B1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2651635, upper bound: 3.3446277
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A1_A1

### Backsubstitution after applying IS history:
0: -0.3749210, 0.5280402, -0.6649656, 1.0442863, -1.4192072, 1.1930058
1: -0.6079230, 0.7211781, -1.0587952, 1.4513346, -2.0592573, 1.7799730
2: -0.4150336, 0.7941813, -0.7051855, 1.5741069, -1.9891405, 1.4993669
3: -0.9699949, 0.9667887, -1.8298885, 1.9018518, -2.8718467, 2.7966771
4: -0.5988675, 1.0718957, -1.1037514, 2.0779939, -2.6768613, 2.1756473

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B1_A2_A1_A1_B1

### Relational analysis result of IS_A1_B2_B1_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2623372, upper bound: 3.2795955
time: 0.41 seconds

## Relational analysis of IS_A1_B2_B1_A2_A1_A1_B2

### Relational analysis result of IS_A1_B2_B1_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2528733, upper bound: 3.2737421
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A1_A2

### Backsubstitution after applying IS history:
0: -0.5635077, 0.8175728, -0.7934000, 1.3059156, -1.8694228, 1.6109729
1: -0.9021713, 1.1278530, -1.2584804, 1.8233461, -2.7255173, 2.3863330
2: -0.5956552, 1.2551364, -0.8515308, 1.9400653, -2.5357196, 2.1066670
3: -1.5388986, 1.5011146, -2.1815679, 2.3729606, -3.9118578, 3.6826818
4: -0.9107077, 1.6853819, -1.3470855, 2.5378199, -3.4485266, 3.0324669

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_B1_A2_A1_A2_B1

### Relational analysis result of IS_A1_B2_B1_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2977522, upper bound: 3.3271636
time: 0.44 seconds

## Relational analysis of IS_A1_B2_B1_A2_A1_A2_B2

### Relational analysis result of IS_A1_B2_B1_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2977522, upper bound: 3.3271636
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.6416510, 0.9915570, -0.1827558, 0.1772476, -0.8188984, 1.1743128
1: -1.0385040, 1.4355035, -0.2525205, 0.2537508, -1.2922548, 1.6880240
2: -0.6867346, 1.4775138, -0.2220088, 0.2736181, -0.9603527, 1.6995225
3: -1.7519407, 1.8813579, -0.3375506, 0.3021557, -2.0540965, 2.2189085
4: -1.0710213, 1.9569094, -0.2093953, 0.3411161, -1.4121374, 2.1663048

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B1_A2_A2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2512497, upper bound: 3.2690055
time: 0.42 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2512497, upper bound: 3.3086897
time: 0.46 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.5382956, 0.7782580, -0.5810655, 0.8817999, -1.4200956, 1.3593235
1: -0.8745649, 1.1203797, -0.9400832, 1.2140620, -2.0886269, 2.0604625
2: -0.5755966, 1.1776516, -0.6256140, 1.3549601, -1.9305567, 1.8032652
3: -1.4538289, 1.4902047, -1.6274993, 1.5925598, -3.0463886, 3.1177037
4: -0.8802605, 1.5773512, -0.9316587, 1.7966312, -2.6768913, 2.5090098

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B1_A2_A2_B2_A1

### Relational analysis result of IS_A1_B2_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2512497, upper bound: 3.2690055
time: 0.42 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2_B2_A2

### Relational analysis result of IS_A1_B2_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2512497, upper bound: 3.3086896
time: 0.45 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -1.0974813, 1.9598112, -0.4430333, 0.6142763, -1.7117574, 2.4028444
1: -1.7284176, 2.7616868, -0.7312632, 0.8677829, -2.5962005, 3.4929490
2: -1.1917427, 2.8475709, -0.4858184, 0.9329896, -2.1247320, 3.3333893
3: -3.0469224, 3.5289857, -1.1720706, 1.1719851, -4.2189074, 4.7010555
4: -1.9017955, 3.6752985, -0.7177969, 1.2603147, -3.1621103, 4.3930950

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_B2_A1_B1_B1_A1

### Relational analysis result of IS_A1_B2_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3358159, upper bound: 3.3719798
time: 0.44 seconds

## Relational analysis of IS_A1_B2_B2_A1_B1_B1_A2

### Relational analysis result of IS_A1_B2_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3358159, upper bound: 3.3724874
time: 0.45 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.9088031, 1.5758088, -0.3184724, 0.4474630, -1.3562658, 1.8942807
1: -1.4398271, 2.2360628, -0.5316270, 0.6528761, -2.0927031, 2.7676897
2: -0.9780726, 2.3087542, -0.3685142, 0.6536136, -1.6316861, 2.6772683
3: -2.5214007, 2.8758447, -0.7876929, 0.8670435, -3.3884437, 3.6635375
4: -1.5621052, 2.9985361, -0.5156054, 0.8533601, -2.4154654, 3.5141408

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_B2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_B2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B2_A1_B1_B2_B1

### Relational analysis result of IS_A1_B2_B2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2991443, upper bound: 3.3555630
time: 0.44 seconds

## Relational analysis of IS_A1_B2_B2_A1_B1_B2_B2

### Relational analysis result of IS_A1_B2_B2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3203379, upper bound: 3.3616577
time: 0.42 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.9817879, 1.7485526, -0.7697797, 1.2606049, -2.2423923, 2.5183320
1: -1.5599338, 2.4556642, -1.2328054, 1.8236289, -3.3835626, 3.6884689
2: -1.0756251, 2.5230427, -0.8312731, 1.8523769, -2.9280019, 3.3543158
3: -2.7106605, 3.1420727, -2.1008234, 2.3705170, -5.0811777, 5.2428956
4: -1.7095270, 3.2609978, -1.3185239, 2.4256351, -4.1351604, 4.5795207

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_B2_A1_B2_A1_A1

### Relational analysis result of IS_A1_B2_B2_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3391601, upper bound: 3.3709220
time: 0.41 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2_A1_A2

### Relational analysis result of IS_A1_B2_B2_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3402501, upper bound: 3.3739983
time: 0.44 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -1.1742514, 2.1143587, -0.7697797, 1.2606049, -2.4348564, 2.8841383
1: -1.8470625, 2.9789019, -1.2328054, 1.8236289, -3.6706915, 4.2117071
2: -1.2812233, 3.0621445, -0.8312731, 1.8523769, -3.1336002, 3.8934174
3: -3.2553840, 3.7963691, -2.1008234, 2.3705170, -5.6259012, 5.8971910
4: -2.0455332, 3.9394255, -1.3185239, 2.4256351, -4.4711680, 5.2579494

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_B2_A1_B2_A2_A1

### Relational analysis result of IS_A1_B2_B2_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3391601, upper bound: 3.3712851
time: 0.49 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2_A2_A2

### Relational analysis result of IS_A1_B2_B2_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3402501, upper bound: 3.3739983
time: 0.46 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.1723508, 0.1510008, -0.2014111, 0.2331109, -0.4054617, 0.3524119
1: -0.2368947, 0.2176939, -0.2856315, 0.3201790, -0.5570737, 0.5033254
2: -0.2098696, 0.2328565, -0.2480963, 0.3442590, -0.5541286, 0.4809528
3: -0.3114822, 0.2593150, -0.3836168, 0.3987596, -0.7102418, 0.6429318
4: -0.1928091, 0.2867447, -0.2511029, 0.4266263, -0.6194353, 0.5378476

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 39

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_B2_A2_A1_B1_B1

### Relational analysis result of IS_A1_B2_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2809320, upper bound: 3.2790158
time: 0.47 seconds

## Relational analysis of IS_A1_B2_B2_A2_A1_B1_B2

### Relational analysis result of IS_A1_B2_B2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2789343, upper bound: 3.2786748
time: 0.40 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.1723508, 0.1510008, -0.6338164, 0.9953935, -1.1677443, 0.7848171
1: -0.2368947, 0.2176939, -1.0270221, 1.4060383, -1.6429330, 1.2447159
2: -0.2098696, 0.2328565, -0.6786129, 1.5029184, -1.7127879, 0.9114695
3: -0.3114822, 0.2593150, -1.7784388, 1.8282951, -2.1397772, 2.0377536
4: -0.1928091, 0.2867447, -1.0270165, 1.9844714, -2.1772804, 1.3137611

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_B2_A2_A1_B2_B1

### Relational analysis result of IS_A1_B2_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2809320, upper bound: 3.2821644
time: 0.42 seconds

## Relational analysis of IS_A1_B2_B2_A2_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2789343, upper bound: 3.2845114
time: 0.42 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.6521097, 1.0343874, -0.2014111, 0.2331109, -0.8852206, 1.2357985
1: -1.0542829, 1.4470388, -0.2856315, 0.3201790, -1.3744619, 1.7326703
2: -0.6988323, 1.5644706, -0.2480963, 0.3442590, -1.0430913, 1.8125669
3: -1.8297936, 1.8823802, -0.3836168, 0.3987596, -2.2285528, 2.2659969
4: -1.0567658, 2.0690625, -0.2511029, 0.4266263, -1.4833920, 2.3201654

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_B2_A2_A2_B1_A1

### Relational analysis result of IS_A1_B2_B2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2867587, upper bound: 3.3073596
time: 0.43 seconds

## Relational analysis of IS_A1_B2_B2_A2_A2_B1_A2

### Relational analysis result of IS_A1_B2_B2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2851880, upper bound: 3.3009790
time: 0.43 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.6521097, 1.0343874, -0.6338164, 0.9953935, -1.6475029, 1.6682035
1: -1.0542829, 1.4470388, -1.0270221, 1.4060383, -2.4603212, 2.4740610
2: -0.6988323, 1.5644706, -0.6786129, 1.5029184, -2.2017508, 2.2430832
3: -1.8297936, 1.8823802, -1.7784388, 1.8282951, -3.6580887, 3.6608179
4: -1.0567658, 2.0690625, -1.0270165, 1.9844714, -3.0412374, 3.0960789

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_B2_A2_A2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2589131, upper bound: 3.3132489
time: 0.43 seconds

## Relational analysis of IS_A1_B2_B2_A2_A2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2589131, upper bound: 3.3142489
time: 0.47 seconds

## BFS IS instance: IS_A2_B1_A1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -1.6506783, 3.1370323, -0.2617934, 0.3611097, -2.0117879, 3.3988256
1: -2.5993731, 4.3526740, -0.4181154, 0.5044270, -3.1037996, 4.7707896
2: -1.8317239, 4.5088830, -0.3043645, 0.5394268, -2.3711505, 4.8132477
3: -4.6305547, 5.4769077, -0.6211426, 0.6629318, -5.2934866, 6.0980501
4: -2.9450130, 5.7203417, -0.4105456, 0.7025315, -3.6475444, 6.1308870

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B1_B1_B1_B1

### Relational analysis result of IS_A2_B1_A1_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2753917, upper bound: 3.3482456
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A1_B1_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_A1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1_B1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3012832, upper bound: 3.3619203
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A1_B1_B1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3012832, upper bound: 3.3619203
time: 0.44 seconds

## BFS IS instance: IS_A2_B1_A1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -1.6506783, 3.1370323, -0.3951213, 0.5633588, -2.2140372, 3.5321536
1: -2.5993731, 4.3526740, -0.6511518, 0.8090668, -3.4084399, 5.0038257
2: -1.8317239, 4.5088830, -0.4504019, 0.8161148, -2.6478381, 4.9592848
3: -4.6305547, 5.4769077, -0.9868081, 1.0850148, -5.7155695, 6.4637156
4: -2.9450130, 5.7203417, -0.6606873, 1.0882038, -4.0332160, 6.3810287

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B1_B1_B2_B1

### Relational analysis result of IS_A2_B1_A1_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2753917, upper bound: 3.3482456
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A1_B1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A1_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_A1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_A1_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1_B1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3012832, upper bound: 3.3619203
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A1_B1_B1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3012832, upper bound: 3.3619203
time: 0.44 seconds

## BFS IS instance: IS_A2_B1_A1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -1.4304782, 2.6815753, -0.0699276, 0.0282996, -1.4587778, 2.7515030
1: -2.2607698, 3.7439203, -0.0797520, 0.0487629, -2.3095326, 3.8236723
2: -1.5827039, 3.8662219, -0.0662054, 0.0409896, -1.6236936, 3.9324274
3: -4.0128703, 4.7290335, -0.0757139, 0.0568801, -4.0697503, 4.8047476
4: -2.5427520, 4.9211960, -0.0569549, 0.0563383, -2.5990901, 4.9781508

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A1_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_A1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_A1_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A1_B1_B2_B1_B1

### Relational analysis result of IS_A2_B1_A1_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2741991, upper bound: 3.3481102
time: 0.45 seconds

## Relational analysis of IS_A2_B1_A1_B1_B2_B1_B2

### Relational analysis result of IS_A2_B1_A1_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2741991, upper bound: 3.3481102
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -1.2784927, 2.3706379, -0.2149342, 0.2436338, -1.5221264, 2.5855720
1: -2.0259395, 3.3208127, -0.3023222, 0.3381470, -2.3640864, 3.6231349
2: -1.4116800, 3.4298153, -0.2634902, 0.3557731, -1.7674531, 3.6933055
3: -3.5872304, 4.2097073, -0.3977464, 0.4134007, -4.0006313, 4.6074538
4: -2.2677824, 4.3767452, -0.2584647, 0.4357610, -2.7035434, 4.6352100

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A1_B1_B2_B2_B1

### Relational analysis result of IS_A2_B1_A1_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2655492, upper bound: 3.3451613
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A1_B1_B2_B2_B2

### Relational analysis result of IS_A2_B1_A1_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2655492, upper bound: 3.3451613
time: 0.44 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -1.5560662, 2.9602537, -0.3464906, 0.4711956, -2.0272617, 3.3067443
1: -2.4622838, 4.1165972, -0.5599223, 0.6732408, -3.1355245, 4.6765189
2: -1.7381214, 4.2417507, -0.3868120, 0.6955789, -2.4337003, 4.6285625
3: -4.3564076, 5.1875057, -0.8629898, 0.8969156, -5.2533231, 6.0504942
4: -2.7980089, 5.3816228, -0.5574431, 0.9282498, -3.7262588, 5.9390659

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2505700, upper bound: 3.3465948
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2770796, upper bound: 3.3603196
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2770796, upper bound: 3.3603196
time: 0.47 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -1.3024096, 2.4356272, -0.3131460, 0.4183234, -1.7207330, 2.7487733
1: -2.0697458, 3.4181411, -0.5029595, 0.5940718, -2.6638176, 3.9211004
2: -1.4510185, 3.5019073, -0.3534136, 0.6200035, -2.0710220, 3.8553205
3: -3.6427433, 4.3293533, -0.7618263, 0.7885701, -4.4313121, 5.0911789
4: -2.3369973, 4.4539409, -0.4942293, 0.8198737, -3.1568708, 4.9481702

Time for backsubstitution: 2.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2740240, upper bound: 3.3537394
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2661041, upper bound: 3.3476444
time: 0.44 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -1.7361624, 3.3105764, -0.3464906, 0.4711956, -2.2073579, 3.6570671
1: -2.7317224, 4.5912924, -0.5599223, 0.6732408, -3.4049630, 5.1512146
2: -1.9310199, 4.7516222, -0.3868120, 0.6955789, -2.6265988, 5.1384344
3: -4.8668809, 5.7745323, -0.8629898, 0.8969156, -5.7637963, 6.6375213
4: -3.1064777, 6.0211258, -0.5574431, 0.9282498, -4.0347276, 6.5785689

Time for backsubstitution: 2.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2497037, upper bound: 3.3409241
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2763470, upper bound: 3.3551089
time: 0.45 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2763470, upper bound: 3.3551089
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -1.5083258, 2.8344274, -0.3131460, 0.4183234, -1.9266491, 3.1475732
1: -2.3805206, 3.9562616, -0.5029595, 0.5940718, -2.9745924, 4.4592204
2: -1.6723577, 4.0783739, -0.3534136, 0.6200035, -2.2923610, 4.4317865
3: -4.2223072, 4.9944415, -0.7618263, 0.7885701, -5.0108762, 5.7562675
4: -2.6861169, 5.1843133, -0.4942293, 0.8198737, -3.5059905, 5.6785426

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2731498, upper bound: 3.3479841
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2655492, upper bound: 3.3450352
time: 0.44 seconds

## BFS IS instance: IS_A2_B1_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.7295296, 1.1918771, -0.3505423, 0.4785340, -1.2080634, 1.5424194
1: -1.1803851, 1.6525177, -0.5681625, 0.6784940, -1.8588791, 2.2206802
2: -0.7948850, 1.7709039, -0.3916546, 0.7108517, -1.5057367, 2.1625581
3: -2.0109479, 2.1634738, -0.8819222, 0.9062943, -2.9172421, 3.0453961
4: -1.2450850, 2.3099849, -0.5643402, 0.9507880, -2.1958730, 2.8743253

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2185432, upper bound: 3.2302957
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_A2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_A1_B1_A1_A1

### Relational analysis result of IS_A2_B1_A2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2552621, upper bound: 3.2722946
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_A1_B1_A1_A2

### Relational analysis result of IS_A2_B1_A2_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2552621, upper bound: 3.2722946
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.9672658, 1.6615082, -0.4331041, 0.5982436, -1.5655094, 2.0946124
1: -1.5381134, 2.2955647, -0.7036965, 0.8493544, -2.3874679, 2.9992609
2: -1.0526744, 2.4516878, -0.4736690, 0.9001736, -1.9528476, 2.9253569
3: -2.6789579, 2.9714186, -1.1340095, 1.1420630, -3.8210211, 4.1054282
4: -1.6691730, 3.1776254, -0.7076730, 1.2180282, -2.8872008, 3.8852983

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2553438, upper bound: 3.2715547
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2553438, upper bound: 3.2715547
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.2172135, 0.2739394, -0.3818156, 0.5175719, -0.7347854, 0.6557550
1: -0.3264682, 0.3846606, -0.6215257, 0.7316089, -1.0580771, 1.0061864
2: -0.2694253, 0.4095533, -0.4241971, 0.7762507, -1.0456760, 0.8337505
3: -0.4594626, 0.4842959, -0.9805481, 0.9820492, -1.4415118, 1.4648440
4: -0.2999950, 0.5168065, -0.6143144, 1.0468788, -1.3468738, 1.1311209

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 10

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2365779, upper bound: 3.2365779
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2365779, upper bound: 3.2365779
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.6156306, 0.9630716, -0.3106939, 0.4195822, -1.0352129, 1.2737656
1: -1.0118906, 1.3217546, -0.4991345, 0.5893351, -1.6012257, 1.8208890
2: -0.6774138, 1.4590747, -0.3531555, 0.6256535, -1.3030673, 1.8122302
3: -1.7357610, 1.7323344, -0.7613440, 0.7809713, -2.5167322, 2.4936781
4: -1.0065820, 1.9185852, -0.4887021, 0.8262410, -1.8328230, 2.4072869

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2365779, upper bound: 3.2365779
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2365779, upper bound: 3.2365779
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.8299519, 1.3831671, -0.4331041, 0.5982436, -1.4281955, 1.8162713
1: -1.3454983, 1.9642149, -0.7036965, 0.8493544, -2.1948528, 2.6679106
2: -0.9183186, 2.0372872, -0.4736690, 0.9001736, -1.8184922, 2.5109563
3: -2.2943082, 2.5653307, -1.1340095, 1.1420630, -3.4363713, 3.6993403
4: -1.4502993, 2.6386447, -0.7076730, 1.2180282, -2.6683271, 3.3463173

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2589782, upper bound: 3.2870780
time: 0.44 seconds

## Relational analysis of IS_A2_B1_A2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_A2_B1_A1_A1

### Relational analysis result of IS_A2_B1_A2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2586002, upper bound: 3.2860256
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A2_A2_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2889473, upper bound: 3.3161012
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2889472, upper bound: 3.3161012
time: 0.46 seconds

## BFS IS instance: IS_A2_B1_A2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.5761482, 0.8804262, -0.4331041, 0.5982436, -1.1743917, 1.3135302
1: -0.9586228, 1.2681425, -0.7036965, 0.8493544, -1.8079772, 1.9718382
2: -0.6303466, 1.3166039, -0.4736690, 0.9001736, -1.5305202, 1.7902728
3: -1.6007009, 1.6725934, -1.1340095, 1.1420630, -2.7427640, 2.8066027
4: -0.9653434, 1.7428689, -0.7076730, 1.2180282, -2.1833713, 2.4505415

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2589782, upper bound: 3.2931419
time: 0.51 seconds

## Relational analysis of IS_A2_B1_A2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_A2_B1_A2_A1

### Relational analysis result of IS_A2_B1_A2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2586002, upper bound: 3.3125922
time: 0.49 seconds

## Relational analysis of IS_A2_B1_A2_A2_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2889473, upper bound: 3.3197698
time: 0.45 seconds

## Relational analysis of IS_A2_B1_A2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2889472, upper bound: 3.3197698
time: 0.45 seconds

## BFS IS instance: IS_A2_B1_A2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.2765033, 0.3798913, -0.3818156, 0.5175719, -0.7940753, 0.7617069
1: -0.4442960, 0.5225174, -0.6215257, 0.7316089, -1.1759049, 1.1440431
2: -0.3321727, 0.5629875, -0.4241971, 0.7762507, -1.1084235, 0.9871846
3: -0.6495922, 0.6825720, -0.9805481, 0.9820492, -1.6316414, 1.6631200
4: -0.4088174, 0.7229737, -0.6143144, 1.0468788, -1.4556961, 1.3372881

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2422171, upper bound: 3.2585734
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2422171, upper bound: 3.2585734
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.7299656, 1.1985065, -0.3106939, 0.4195822, -1.1495478, 1.5092003
1: -1.2001468, 1.6867248, -0.4991345, 0.5893351, -1.7894819, 2.1858587
2: -0.8062264, 1.7816978, -0.3531555, 0.6256535, -1.4318799, 2.1348531
3: -2.0552711, 2.1951113, -0.7613440, 0.7809713, -2.8362420, 2.9564552
4: -1.2266954, 2.3119917, -0.4887021, 0.8262410, -2.0529363, 2.8006935

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2522087, upper bound: 3.2919245
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2522087, upper bound: 3.2919245
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -1.5499620, 2.9211068, -0.4279954, 0.5938456, -2.1438076, 3.3491020
1: -2.4442379, 4.0590744, -0.7078946, 0.8342253, -3.2784631, 4.7669687
2: -1.7187898, 4.2115712, -0.4718082, 0.8979282, -2.6167176, 4.6833792
3: -4.3464975, 5.1185570, -1.1286074, 1.1275156, -5.4740133, 6.2471642
4: -2.7597313, 5.3520274, -0.6930985, 1.2122529, -3.9719841, 6.0451250

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B1_B1_B1_B1

### Relational analysis result of IS_A2_B2_A1_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2965572, upper bound: 3.3540143
time: 0.44 seconds

## Relational analysis of IS_A2_B2_A1_B1_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B2_A1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_B1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3228066, upper bound: 3.3672342
time: 0.45 seconds

## Relational analysis of IS_A2_B2_A1_B1_B1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3228066, upper bound: 3.3672342
time: 0.46 seconds

## BFS IS instance: IS_A2_B2_A1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -1.5499620, 2.9211068, -0.5932168, 0.8270012, -2.3769631, 3.5143225
1: -2.4442379, 4.0590744, -0.9848863, 1.1912254, -3.6354630, 5.0439606
2: -1.7187898, 4.2115712, -0.6512032, 1.2658960, -2.9846854, 4.8627744
3: -4.3464975, 5.1185570, -1.6120055, 1.6211529, -5.9676504, 6.7305622
4: -2.7597313, 5.3520274, -0.9968907, 1.7319027, -4.4916339, 6.3489180

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B1_B1_B2_B1

### Relational analysis result of IS_A2_B2_A1_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2965572, upper bound: 3.3540143
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A1_B1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B2_A1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A1_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_B1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3228066, upper bound: 3.3672342
time: 0.44 seconds

## Relational analysis of IS_A2_B2_A1_B1_B1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3228066, upper bound: 3.3672342
time: 0.44 seconds

## BFS IS instance: IS_A2_B2_A1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -1.3322984, 2.4751472, -0.1171788, 0.0416433, -1.3739417, 2.5923259
1: -2.1097126, 3.4620111, -0.1711728, 0.0606157, -2.1703284, 3.6331840
2: -1.4727931, 3.5772123, -0.1402625, 0.0546926, -1.5274856, 3.7174749
3: -3.7357616, 4.3850961, -0.1835253, 0.0704479, -3.8062088, 4.5686212
4: -2.3652546, 4.5634842, -0.0573432, 0.0723072, -2.4375618, 4.6208272

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B2_A1_B1_B2_B1_B1

### Relational analysis result of IS_A2_B2_A1_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2991403, upper bound: 3.3548499
time: 0.43 seconds

## Relational analysis of IS_A2_B2_A1_B1_B2_B1_B2

### Relational analysis result of IS_A2_B2_A1_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2991403, upper bound: 3.3548499
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -1.1844230, 2.1715832, -0.2262682, 0.2995241, -1.4839470, 2.3978515
1: -1.8815413, 3.0478215, -0.3343897, 0.4018024, -2.2833438, 3.3822112
2: -1.3058197, 3.1492691, -0.2806285, 0.4328869, -1.7387067, 3.4298975
3: -3.3195398, 3.8765244, -0.4653418, 0.5069168, -3.8264561, 4.3418665
4: -2.0962961, 4.0282106, -0.2887270, 0.5376805, -2.6339765, 4.3169374

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B2_A1_B1_B2_B2_B1

### Relational analysis result of IS_A2_B2_A1_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3206301, upper bound: 3.3614289
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A1_B1_B2_B2_B2

### Relational analysis result of IS_A2_B2_A1_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3206301, upper bound: 3.3614289
time: 0.49 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -1.4567093, 2.7474623, -1.0188239, 1.7740973, -3.2308056, 3.7662857
1: -2.3098607, 3.8269162, -1.6081145, 2.5438516, -4.8537121, 5.4350305
2: -1.6270657, 3.9458733, -1.1081879, 2.5785356, -4.2056007, 5.0540609
3: -4.0754924, 4.8336587, -2.8083901, 3.2681413, -7.3436332, 7.6420479
4: -2.6146660, 5.0151505, -1.7669487, 3.3312380, -5.9459033, 6.7820983

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3482370, upper bound: 3.3830199
time: 0.48 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3482370, upper bound: 3.3830199
time: 0.44 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 9.39 seconds
IS_A1_B2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.2996012, upper bound: 3.3626767
IS_A1_B2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.2996012, upper bound: 3.3626767
IS_A1_B2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.2735803, upper bound: 3.3487259
IS_A1_B2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.2651580, upper bound: 3.3447562
IS_A1_B2_B1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.2754982, upper bound: 3.3532808
IS_A1_B2_B1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.2754982, upper bound: 3.3563685
IS_A1_B2_B1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.2726117, upper bound: 3.3485057
IS_A1_B2_B1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.2651635, upper bound: 3.3446277
IS_A1_B2_B1_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.2623372, upper bound: 3.2795955
IS_A1_B2_B1_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.2528733, upper bound: 3.2737421
IS_A1_B2_B1_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.2977522, upper bound: 3.3271636
IS_A1_B2_B1_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.2977522, upper bound: 3.3271636
IS_A1_B2_B1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.2512497, upper bound: 3.2690055
IS_A1_B2_B1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.2512497, upper bound: 3.3086897
IS_A1_B2_B1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.2512497, upper bound: 3.2690055
IS_A1_B2_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.2512497, upper bound: 3.3086896
IS_A1_B2_B2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.3358159, upper bound: 3.3719798
IS_A1_B2_B2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.3358159, upper bound: 3.3724874
IS_A1_B2_B2_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.2991443, upper bound: 3.3555630
IS_A1_B2_B2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.3203379, upper bound: 3.3616577
IS_A1_B2_B2_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.3391601, upper bound: 3.3709220
IS_A1_B2_B2_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.3402501, upper bound: 3.3739983
IS_A1_B2_B2_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.3391601, upper bound: 3.3712851
IS_A1_B2_B2_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.3402501, upper bound: 3.3739983
IS_A1_B2_B2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.2809320, upper bound: 3.2790158
IS_A1_B2_B2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.2789343, upper bound: 3.2786748
IS_A1_B2_B2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.2809320, upper bound: 3.2821644
IS_A1_B2_B2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.2789343, upper bound: 3.2845114
IS_A1_B2_B2_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.2867587, upper bound: 3.3073596
IS_A1_B2_B2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.2851880, upper bound: 3.3009790
IS_A1_B2_B2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.2589131, upper bound: 3.3132489
IS_A1_B2_B2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.2589131, upper bound: 3.3142489
IS_A2_B1_A1_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.3012832, upper bound: 3.3619203
IS_A2_B1_A1_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.3012832, upper bound: 3.3619203
IS_A2_B1_A1_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.3012832, upper bound: 3.3619203
IS_A2_B1_A1_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.3012832, upper bound: 3.3619203
IS_A2_B1_A1_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.2741991, upper bound: 3.3481102
IS_A2_B1_A1_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.2741991, upper bound: 3.3481102
IS_A2_B1_A1_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.2655492, upper bound: 3.3451613
IS_A2_B1_A1_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.2655492, upper bound: 3.3451613
IS_A2_B1_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.2770796, upper bound: 3.3603196
IS_A2_B1_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.2770796, upper bound: 3.3603196
IS_A2_B1_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.2740240, upper bound: 3.3537394
IS_A2_B1_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.2661041, upper bound: 3.3476444
IS_A2_B1_A1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.2763470, upper bound: 3.3551089
IS_A2_B1_A1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.2763470, upper bound: 3.3551089
IS_A2_B1_A1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.2731498, upper bound: 3.3479841
IS_A2_B1_A1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.2655492, upper bound: 3.3450352
IS_A2_B1_A2_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.2552621, upper bound: 3.2722946
IS_A2_B1_A2_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.2552621, upper bound: 3.2722946
IS_A2_B1_A2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.2553438, upper bound: 3.2715547
IS_A2_B1_A2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.2553438, upper bound: 3.2715547
IS_A2_B1_A2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.2365779, upper bound: 3.2365779
IS_A2_B1_A2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.2365779, upper bound: 3.2365779
IS_A2_B1_A2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.2365779, upper bound: 3.2365779
IS_A2_B1_A2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.2365779, upper bound: 3.2365779
IS_A2_B1_A2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.2889473, upper bound: 3.3161012
IS_A2_B1_A2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.2889472, upper bound: 3.3161012
IS_A2_B1_A2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.2889473, upper bound: 3.3197698
IS_A2_B1_A2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.2889472, upper bound: 3.3197698
IS_A2_B1_A2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.2422171, upper bound: 3.2585734
IS_A2_B1_A2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.2422171, upper bound: 3.2585734
IS_A2_B1_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.2522087, upper bound: 3.2919245
IS_A2_B1_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.2522087, upper bound: 3.2919245
IS_A2_B2_A1_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.3228066, upper bound: 3.3672342
IS_A2_B2_A1_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.3228066, upper bound: 3.3672342
IS_A2_B2_A1_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.3228066, upper bound: 3.3672342
IS_A2_B2_A1_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.3228066, upper bound: 3.3672342
IS_A2_B2_A1_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.2991403, upper bound: 3.3548499
IS_A2_B2_A1_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.2991403, upper bound: 3.3548499
IS_A2_B2_A1_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.3206301, upper bound: 3.3614289
IS_A2_B2_A1_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.3206301, upper bound: 3.3614289
IS_A2_B2_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.3482370, upper bound: 3.3830199
IS_A2_B2_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 9.39
Output dim: 0, lower bound: -3.3482370, upper bound: 3.3830199
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 0, lower bound: -3.3476916, upper bound: 3.3785540
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 0, lower bound: -3.3476916, upper bound: 3.3783250
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 0, lower bound: -3.3476916, upper bound: 3.3783250
IS_A2_B2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 0, lower bound: -3.2824701, upper bound: 3.2806231
IS_A2_B2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 0, lower bound: -3.2801292, upper bound: 3.2801292
IS_A2_B2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 0, lower bound: -3.2824701, upper bound: 3.2837556
IS_A2_B2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 0, lower bound: -3.2801292, upper bound: 3.2856642
IS_A2_B2_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 0, lower bound: -3.2837555, upper bound: 3.2938538
IS_A2_B2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 0, lower bound: -3.2856642, upper bound: 3.2991177
IS_A2_B2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 0, lower bound: -3.2448587, upper bound: 3.2511194
IS_A2_B2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 9.39
Output dim: 0, lower bound: -3.2448587, upper bound: 3.3137138
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=3.5238394737243652
rel_dist={0: [-3.3982329481346, 3.3982329481346003]}

## Binary search (step 2) starts
Candidate diff: 0.0625000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3740686, upper bound: 3.3318377
time: 0.41 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3943702, upper bound: 3.3943701
time: 0.43 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.04 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.04
Output dim: 0, lower bound: -3.3740686, upper bound: 3.3318377
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.04
Output dim: 0, lower bound: -3.3943702, upper bound: 3.3943701

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.2895984, 0.3943615, -0.8808075, 1.5047889, -1.7943872, 1.2751689
1: -0.4578271, 0.5641531, -1.3966537, 2.1583297, -2.6161568, 1.9608067
2: -0.3370600, 0.5814233, -0.9524446, 2.2021837, -2.5392432, 1.5338678
3: -0.6758766, 0.7399411, -2.4135633, 2.7874823, -3.4633589, 3.1535044
4: -0.4528955, 0.7513722, -1.5303535, 2.8573139, -3.3102093, 2.2817256

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3636891, upper bound: 3.3211754
time: 0.41 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3632708, upper bound: 3.3083600
time: 0.43 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -1.0333478, 1.8749698, -0.9835699, 1.7383295, -2.7716773, 2.8585393
1: -1.6493901, 2.6237774, -1.5618739, 2.4618018, -4.1111917, 4.1856508
2: -1.1160958, 2.7230368, -1.0621964, 2.5254564, -3.6415520, 3.7852330
3: -2.9354391, 3.3341267, -2.7349937, 3.1533265, -6.0887647, 6.0691204
4: -1.7549448, 3.5216486, -1.6960667, 3.2828889, -5.0378327, 5.2177153

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3318377, upper bound: 3.3740686
time: 0.43 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3318377, upper bound: 3.3740686
time: 0.43 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.14 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 3.14
Output dim: 0, lower bound: -3.3636891, upper bound: 3.3211754
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 3.14
Output dim: 0, lower bound: -3.3632708, upper bound: 3.3083600
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.14
Output dim: 0, lower bound: -3.3318377, upper bound: 3.3740686
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.14
Output dim: 0, lower bound: -3.3318377, upper bound: 3.3740686

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -0.1909689, 0.2010397, -0.8153566, 1.3730776, -1.5640465, 1.0163963
1: -0.2711346, 0.2899297, -1.2960849, 1.9708279, -2.2419624, 1.5860146
2: -0.2349898, 0.3038281, -0.8794892, 2.0191257, -2.2541156, 1.1833173
3: -0.3588729, 0.3552319, -2.2364023, 2.5519223, -2.9107952, 2.5916340
4: -0.2300957, 0.3767083, -1.4116271, 2.6237235, -2.8538194, 1.7883353

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 39

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3608093, upper bound: 3.3206361
time: 0.43 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3470489, upper bound: 3.3052482
time: 0.40 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -0.1886021, 0.1795645, -0.6167785, 0.9663604, -1.1549625, 0.7963429
1: -0.2620578, 0.2599264, -0.9888538, 1.3913028, -1.6533606, 1.2487803
2: -0.2305741, 0.2687693, -0.6568786, 1.4573338, -1.6879079, 0.9256476
3: -0.3403661, 0.3140820, -1.6944852, 1.8231517, -2.1635177, 2.0085669
4: -0.2184048, 0.3308211, -1.0424186, 1.9161193, -2.1345241, 1.3732394

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_A1

### Relational analysis result of IS_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3384957, upper bound: 3.2831206
time: 0.45 seconds

## Relational analysis of IS_A1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2883998, upper bound: 3.2883998
time: 0.40 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2883998, upper bound: 3.2883998
time: 0.43 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -1.0333478, 1.8749698, -0.2895984, 0.3943615, -1.4277093, 2.1645677
1: -1.6493901, 2.6237774, -0.4578271, 0.5641531, -2.2135432, 3.0816045
2: -1.1160958, 2.7230368, -0.3370600, 0.5814233, -1.6975191, 3.0600965
3: -2.9354391, 3.3341267, -0.6758766, 0.7399411, -3.6753798, 4.0100031
4: -1.7549448, 3.5216486, -0.4528955, 0.7513722, -2.5063167, 3.9745436

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3073190, upper bound: 3.3603359
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3083599, upper bound: 3.3632707
time: 0.41 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -1.0333478, 1.8749698, -1.0299754, 1.8632205, -2.8965681, 2.9049451
1: -1.6493901, 2.6237774, -1.6423873, 2.6173930, -4.2667828, 4.2661643
2: -1.1160958, 2.7230368, -1.1113069, 2.6993732, -3.8154690, 3.8343437
3: -2.9354391, 3.3341267, -2.9127293, 3.3258715, -6.2613106, 6.2468557
4: -1.7549448, 3.5216486, -1.7470773, 3.4881601, -5.2431049, 5.2687259

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3318377, upper bound: 3.3939926
time: 0.46 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3318343, upper bound: 3.3934415
time: 0.42 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.27 seconds
IS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 3.27
Output dim: 0, lower bound: -3.3608093, upper bound: 3.3206361
IS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 3.27
Output dim: 0, lower bound: -3.3470489, upper bound: 3.3052482
IS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 3.27
Output dim: 0, lower bound: -3.2883998, upper bound: 3.2883998
IS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 3.27
Output dim: 0, lower bound: -3.2883998, upper bound: 3.2883998
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.27
Output dim: 0, lower bound: -3.3073190, upper bound: 3.3603359
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.27
Output dim: 0, lower bound: -3.3083599, upper bound: 3.3632707
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.27
Output dim: 0, lower bound: -3.3318377, upper bound: 3.3939926
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.27
Output dim: 0, lower bound: -3.3318343, upper bound: 3.3934415

## BFS IS instance: IS_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1702632, 0.1284280, -0.5612379, 0.8326873, -1.0029504, 0.6896659
1: -0.2304049, 0.1944827, -0.8971887, 1.1994146, -1.4298196, 1.0916713
2: -0.2067248, 0.1976529, -0.5975255, 1.2771761, -1.4839010, 0.7951784
3: -0.2998832, 0.2286490, -1.5237699, 1.5878348, -1.8877180, 1.7524188
4: -0.1890529, 0.2390879, -0.9338441, 1.6894978, -1.8785508, 1.1729319

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A1_B1_A1

### Relational analysis result of IS_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3469762, upper bound: 3.3052324
time: 0.42 seconds

## Relational analysis of IS_A1_A1_B1_A2

### Relational analysis result of IS_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3469762, upper bound: 3.3052482
time: 0.39 seconds

## BFS IS instance: IS_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1729392, 0.1374773, -1.0450571, 1.8653935, -2.0383327, 1.1825342
1: -0.2369139, 0.2041424, -1.6523304, 2.6871493, -2.9240632, 1.8564727
2: -0.2106074, 0.2085033, -1.1242528, 2.7166109, -2.9272184, 1.3327560
3: -0.3071967, 0.2408988, -2.9145761, 3.4359851, -3.7431817, 3.1554749
4: -0.1924484, 0.2513706, -1.8184122, 3.4984517, -3.6909001, 2.0697827

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B2_A1

### Relational analysis result of IS_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3433723, upper bound: 3.2984655
time: 0.40 seconds

## Relational analysis of IS_A1_A1_B2_A2

### Relational analysis result of IS_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3433402, upper bound: 3.2984795
time: 0.40 seconds

## BFS IS instance: IS_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.1886021, 0.1795645, -0.1967524, 0.2248845, -0.4134865, 0.3763169
1: -0.2620578, 0.2599264, -0.2822605, 0.3184986, -0.5805563, 0.5421870
2: -0.2305741, 0.2687693, -0.2420871, 0.3388391, -0.5694133, 0.5108563
3: -0.3403661, 0.3140820, -0.3890238, 0.3965968, -0.7369628, 0.7031058
4: -0.2184048, 0.3308211, -0.2476796, 0.4248878, -0.6432926, 0.5785007

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_A2_B1_B1

### Relational analysis result of IS_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2883998, upper bound: 3.2883998
time: 0.47 seconds

## Relational analysis of IS_A1_A2_B1_B2

### Relational analysis result of IS_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2883998, upper bound: 3.2883998
time: 0.44 seconds

## BFS IS instance: IS_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.1886021, 0.1795645, -0.6482157, 1.0445390, -1.2331411, 0.8277801
1: -0.2620578, 0.2599264, -1.0498279, 1.4843211, -1.7463789, 1.3097544
2: -0.2305741, 0.2687693, -0.6974155, 1.5761452, -1.8067193, 0.9661847
3: -0.3403661, 0.3140820, -1.8218029, 1.9235181, -2.2638841, 2.1358840
4: -0.2184048, 0.3308211, -1.0683334, 2.0604343, -2.2788391, 1.3991544

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_A2_B2_B1

### Relational analysis result of IS_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2867515, upper bound: 3.3081144
time: 0.56 seconds

## Relational analysis of IS_A1_A2_B2_B2

### Relational analysis result of IS_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2883373, upper bound: 3.2883373
time: 0.45 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.6167870, 0.9804183, -0.2565626, 0.3443241, -0.9611109, 1.2369809
1: -1.0047641, 1.3907540, -0.4032328, 0.4937120, -1.4984761, 1.7939868
2: -0.6674653, 1.4859817, -0.3054087, 0.5038292, -1.1712945, 1.7913904
3: -1.7353394, 1.8086083, -0.5756149, 0.6428469, -2.3781862, 2.3842232
4: -1.0161959, 1.9481347, -0.3938302, 0.6436344, -1.6598303, 2.3419650

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2841814, upper bound: 3.3404837
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_A1

### Relational analysis result of IS_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2798246, upper bound: 3.2869390
time: 0.44 seconds

## Relational analysis of IS_A2_B1_A1_A2

### Relational analysis result of IS_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3019700, upper bound: 3.3575497
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.5555521, 0.8390003, -0.1967524, 0.2248845, -0.7804365, 1.0357528
1: -0.9075407, 1.1878233, -0.2822605, 0.3184986, -1.2260391, 1.4700838
2: -0.6072665, 1.2878169, -0.2420871, 0.3388391, -0.9461055, 1.5299039
3: -1.5513695, 1.5564189, -0.3890238, 0.3965968, -1.9479663, 1.9454427
4: -0.8984371, 1.7021952, -0.2476796, 0.4248878, -1.3233249, 1.9498748

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2831206, upper bound: 3.3384956
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3083599, upper bound: 3.3632707
time: 0.46 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3083599, upper bound: 3.3632707
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.7388377, 1.2283087, -0.8395992, 1.4494436, -2.1882813, 2.0679078
1: -1.1850184, 1.7239280, -1.3434677, 2.0349193, -3.2199376, 3.0673954
2: -0.7881981, 1.8336043, -0.8949960, 2.1359351, -2.9241328, 2.7285998
3: -2.0788705, 2.2223623, -2.3733349, 2.6056395, -4.6845102, 4.5956969
4: -1.2122128, 2.4006119, -1.3908591, 2.7848406, -3.9970534, 3.7914703

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3670806, upper bound: 3.3811898
time: 0.45 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3863135, upper bound: 3.3873987
time: 0.45 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -1.2271883, 2.3155425, -0.7695433, 1.3076057, -2.5347939, 3.0850856
1: -1.9551189, 3.2611241, -1.2455037, 1.8607799, -3.8158989, 4.5066280
2: -1.3333057, 3.3333139, -0.8279764, 1.9259086, -3.2592142, 4.1612902
3: -3.5210397, 4.1069036, -2.1747916, 2.3906178, -5.9116573, 6.2816949
4: -2.1148672, 4.2876000, -1.2848178, 2.5215352, -4.6364017, 5.5724173

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A2_A1

### Relational analysis result of IS_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3771733, upper bound: 3.3703173
time: 0.45 seconds

## Relational analysis of IS_A2_B2_A2_A2

### Relational analysis result of IS_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3861918, upper bound: 3.3861917
time: 0.46 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.18 seconds
IS_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 0, lower bound: -3.3469762, upper bound: 3.3052324
IS_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 0, lower bound: -3.3469762, upper bound: 3.3052482
IS_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 0, lower bound: -3.3433723, upper bound: 3.2984655
IS_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 0, lower bound: -3.3433402, upper bound: 3.2984795
IS_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 0, lower bound: -3.2883998, upper bound: 3.2883998
IS_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 0, lower bound: -3.2883998, upper bound: 3.2883998
IS_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 0, lower bound: -3.2867515, upper bound: 3.3081144
IS_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 0, lower bound: -3.2883373, upper bound: 3.2883373
IS_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 0, lower bound: -3.2798246, upper bound: 3.2869390
IS_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 0, lower bound: -3.3019700, upper bound: 3.3575497
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 0, lower bound: -3.3083599, upper bound: 3.3632707
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 0, lower bound: -3.3083599, upper bound: 3.3632707
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 0, lower bound: -3.3670806, upper bound: 3.3811898
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 0, lower bound: -3.3863135, upper bound: 3.3873987
IS_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 0, lower bound: -3.3771733, upper bound: 3.3703173
IS_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 0, lower bound: -3.3861918, upper bound: 3.3861917

## BFS IS instance: IS_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1630524, 0.1018633, -0.5612379, 0.8326873, -0.9957396, 0.6631012
1: -0.2162702, 0.1595190, -0.8971887, 1.1994146, -1.4156848, 1.0567076
2: -0.1968231, 0.1582093, -0.5975255, 1.2771761, -1.4739993, 0.7557349
3: -0.2805830, 0.1871109, -1.5237699, 1.5878348, -1.8684179, 1.7108806
4: -0.1795197, 0.1863293, -0.9338441, 1.6894978, -1.8690176, 1.1201735

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_A1_B1_A1_A1

### Relational analysis result of IS_A1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3389717, upper bound: 3.2939083
time: 0.41 seconds

## Relational analysis of IS_A1_A1_B1_A1_A2

### Relational analysis result of IS_A1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3364705, upper bound: 3.2853451
time: 0.41 seconds

## BFS IS instance: IS_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.2088245, 0.2758393, -0.5612379, 0.8326873, -1.0415118, 0.8370771
1: -0.3110615, 0.3673706, -0.8971887, 1.1994146, -1.5104761, 1.2645591
2: -0.2585255, 0.3915462, -0.5975255, 1.2771761, -1.5357016, 0.9890717
3: -0.4219261, 0.4455681, -1.5237699, 1.5878348, -2.0097609, 1.9693379
4: -0.2464714, 0.4945604, -0.9338441, 1.6894978, -1.9359692, 1.4284046

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_A1_B1_A2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2643254, upper bound: 3.2713582
time: 0.44 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2

### Relational analysis result of IS_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2643254, upper bound: 3.3206361
time: 0.42 seconds

## BFS IS instance: IS_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.1490553, 0.0597873, -0.9906490, 1.7536597, -1.9027151, 1.0504364
1: -0.2050755, 0.1007832, -1.5676980, 2.5220642, -2.7271397, 1.6684811
2: -0.1915288, 0.0823605, -1.0632721, 2.5586424, -2.7501712, 1.1456325
3: -0.1981074, 0.1277347, -2.7638965, 3.2295513, -3.4276588, 2.8916311
4: -0.1202420, 0.1087260, -1.7178493, 3.3020234, -3.4222655, 1.8265753

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_A1_B2_A1_A1

### Relational analysis result of IS_A1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3359723, upper bound: 3.2908741
time: 0.40 seconds

## Relational analysis of IS_A1_A1_B2_A1_A2

### Relational analysis result of IS_A1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3359908, upper bound: 3.2906396
time: 0.45 seconds

## BFS IS instance: IS_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.1668644, 0.1055425, -0.8719853, 1.5151944, -1.6820588, 0.9775277
1: -0.2214651, 0.1590646, -1.3874525, 2.1853328, -2.4067979, 1.5465171
2: -0.2022947, 0.1508963, -0.9315935, 2.2238195, -2.4261141, 1.0824897
3: -0.2796538, 0.1883675, -2.4344785, 2.8069587, -3.0866125, 2.6228459
4: -0.1861155, 0.1732125, -1.5067275, 2.8823516, -3.0684671, 1.6799397

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 39

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A1_B2_A2_A1

### Relational analysis result of IS_A1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3433368, upper bound: 3.2984795
time: 0.43 seconds

## Relational analysis of IS_A1_A1_B2_A2_A2

### Relational analysis result of IS_A1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3433368, upper bound: 3.2984795
time: 0.42 seconds

## BFS IS instance: IS_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.1886021, 0.1795645, -0.1909689, 0.2010397, -0.3896418, 0.3705333
1: -0.2620578, 0.2599264, -0.2711346, 0.2899297, -0.5519875, 0.5310610
2: -0.2305741, 0.2687693, -0.2349898, 0.3038281, -0.5344023, 0.5037591
3: -0.3403661, 0.3140820, -0.3588729, 0.3552319, -0.6955979, 0.6729549
4: -0.2184048, 0.3308211, -0.2300957, 0.3767083, -0.5951130, 0.5609168

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_B1_B1_A1

### Relational analysis result of IS_A1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2832122, upper bound: 3.2859375
time: 0.41 seconds

## Relational analysis of IS_A1_A2_B1_B1_A2

### Relational analysis result of IS_A1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2827588, upper bound: 3.2827588
time: 0.39 seconds

## BFS IS instance: IS_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.1886021, 0.1795645, -0.1886021, 0.1795645, -0.3681666, 0.3681666
1: -0.2620578, 0.2599264, -0.2620578, 0.2599264, -0.5219842, 0.5219842
2: -0.2305741, 0.2687693, -0.2305741, 0.2687693, -0.4993434, 0.4993434
3: -0.3403661, 0.3140820, -0.3403661, 0.3140820, -0.6544481, 0.6544481
4: -0.2184048, 0.3308211, -0.2184048, 0.3308211, -0.5492259, 0.5492259

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A1

### Relational analysis result of IS_A1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2736932, upper bound: 3.2620216
time: 0.43 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2

### Relational analysis result of IS_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2826483, upper bound: 3.2826483
time: 0.42 seconds

## BFS IS instance: IS_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.1886021, 0.1795645, -0.5938034, 0.9229457, -1.1115477, 0.7733679
1: -0.2620578, 0.2599264, -0.9660431, 1.3092268, -1.5712845, 1.2259696
2: -0.2305741, 0.2687693, -0.6431762, 1.4069108, -1.6374849, 0.9119455
3: -0.3403661, 0.3140820, -1.6586674, 1.7083746, -2.0487406, 1.9727494
4: -0.2184048, 0.3308211, -0.9717162, 1.8496504, -2.0680552, 1.3025372

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_B2_B1_A1

### Relational analysis result of IS_A1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3592317, upper bound: 3.3072287
time: 0.42 seconds

## Relational analysis of IS_A1_A2_B2_B1_A2

### Relational analysis result of IS_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3540666, upper bound: 3.2938665
time: 0.43 seconds

## BFS IS instance: IS_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.1770213, 0.1356121, -0.7519640, 1.3071438, -1.4841651, 0.8875760
1: -0.2398114, 0.2014719, -1.2321365, 1.8425138, -2.0823252, 1.4336083
2: -0.2142571, 0.2061213, -0.8193258, 1.9329081, -2.1471651, 1.0254471
3: -0.3093022, 0.2367717, -2.1599669, 2.3678660, -2.6771684, 2.3967385
4: -0.1952045, 0.2494673, -1.2762941, 2.4919331, -2.6871376, 1.5257615

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_B2_B2_A1

### Relational analysis result of IS_A1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3349277, upper bound: 3.2700954
time: 0.43 seconds

## Relational analysis of IS_A1_A2_B2_B2_A2

### Relational analysis result of IS_A1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3622409, upper bound: 3.3034641
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.3407221, 0.4752711, -0.2338697, 0.3078303, -0.6485524, 0.7091408
1: -0.5564522, 0.6531985, -0.3620735, 0.4391946, -0.9956468, 1.0152720
2: -0.3940860, 0.7215068, -0.2818983, 0.4530277, -0.8471136, 1.0034051
3: -0.8718258, 0.8605914, -0.5101972, 0.5666184, -1.4384441, 1.3707886
4: -0.5247028, 0.9477615, -0.3489375, 0.5738655, -1.0985680, 1.2966990

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_A1_B1

### Relational analysis result of IS_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2680622, upper bound: 3.2869390
time: 0.46 seconds

## Relational analysis of IS_A2_B1_A1_A1_B2

### Relational analysis result of IS_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2680622, upper bound: 3.2869390
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.4084996, 0.5693522, -0.1883008, 0.2152189, -0.6237184, 0.7576531
1: -0.6703718, 0.7839134, -0.2700287, 0.3022276, -0.9725993, 1.0539421
2: -0.4531251, 0.8762918, -0.2314827, 0.3259848, -0.7791096, 1.1077745
3: -1.0841039, 1.0413346, -0.3718146, 0.3748229, -1.4589268, 1.4131492
4: -0.6257645, 1.1637021, -0.2349494, 0.4087547, -1.0345191, 1.3986516

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 39

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_A2_B1

### Relational analysis result of IS_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2825229, upper bound: 3.3441693
time: 0.45 seconds

## Relational analysis of IS_A2_B1_A1_A2_B2

### Relational analysis result of IS_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2825229, upper bound: 3.3575497
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.5555521, 0.8390003, -0.1909689, 0.2010397, -0.7565917, 1.0299692
1: -0.9075407, 1.1878233, -0.2711346, 0.2899297, -1.1974704, 1.4589579
2: -0.6072665, 1.2878169, -0.2349898, 0.3038281, -0.9110945, 1.5228066
3: -1.5513695, 1.5564189, -0.3588729, 0.3552319, -1.9066014, 1.9152918
4: -0.8984371, 1.7021952, -0.2300957, 0.3767083, -1.2751453, 1.9322909

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_B1

### Relational analysis result of IS_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2831206, upper bound: 3.3384956
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2691295, upper bound: 3.2677982
time: 0.49 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3034837, upper bound: 3.3623153
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.5555521, 0.8390003, -0.1886021, 0.1795645, -0.7351165, 1.0276024
1: -0.9075407, 1.1878233, -0.2620578, 0.2599264, -1.1674671, 1.4498811
2: -0.6072665, 1.2878169, -0.2305741, 0.2687693, -0.8760356, 1.5183910
3: -1.5513695, 1.5564189, -0.3403661, 0.3140820, -1.8654516, 1.8967850
4: -0.8984371, 1.7021952, -0.2184048, 0.3308211, -1.2292582, 1.9205999

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_B1

### Relational analysis result of IS_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2831206, upper bound: 3.3384956
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2691295, upper bound: 3.2677982
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3034837, upper bound: 3.3623154
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.6870305, 1.1130922, -0.4892117, 0.7003186, -1.3873491, 1.6023037
1: -1.1032165, 1.5576646, -0.7909114, 0.9510068, -2.0542233, 2.3485761
2: -0.7334036, 1.6774623, -0.5302792, 1.0910958, -1.8244994, 2.2077415
3: -1.9291157, 2.0173697, -1.3417283, 1.2634159, -3.1925309, 3.3590980
4: -1.1187752, 2.2027237, -0.7672535, 1.4652178, -2.5839925, 2.9699769

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B1_B1

### Relational analysis result of IS_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3345147, upper bound: 3.3576462
time: 0.45 seconds

## Relational analysis of IS_A2_B2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3612467, upper bound: 3.3590332
time: 0.49 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3612467, upper bound: 3.3811898
time: 0.46 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.5287228, 0.7700919, -0.5273240, 0.7683397, -1.2970620, 1.2974157
1: -0.8562238, 1.0700272, -0.8527137, 1.0711391, -1.9273626, 1.9227406
2: -0.5671958, 1.2077169, -0.5645064, 1.1881945, -1.7553903, 1.7722232
3: -1.4637495, 1.4193766, -1.4562628, 1.4143211, -2.8780701, 2.8756394
4: -0.8397263, 1.6099051, -0.8304668, 1.5839341, -2.4236603, 2.4403715

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B2_B1

### Relational analysis result of IS_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3742169, upper bound: 3.3732959
time: 0.45 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2

### Relational analysis result of IS_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3610161, upper bound: 3.3660411
time: 0.44 seconds

## BFS IS instance: IS_A2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.7778159, 1.3267192, -0.7172917, 1.1912274, -1.9690430, 2.0440106
1: -1.2532948, 1.8665893, -1.1632739, 1.6913042, -2.9445987, 3.0298631
2: -0.8307147, 1.9636376, -0.7726474, 1.7680702, -2.5987847, 2.7362850
3: -2.2206659, 2.3894682, -2.0241988, 2.1814270, -4.4020929, 4.4136667
4: -1.2831259, 2.5623760, -1.1894554, 2.3212271, -3.6043530, 3.7518311

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_A1_B1

### Relational analysis result of IS_A2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3743381, upper bound: 3.3657926
time: 0.45 seconds

## Relational analysis of IS_A2_B2_A2_A1_B2

### Relational analysis result of IS_A2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3743381, upper bound: 3.3703173
time: 0.49 seconds

## BFS IS instance: IS_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -1.0385535, 1.8943465, -0.5589097, 0.8379070, -1.8764606, 2.4532559
1: -1.6589499, 2.6707377, -0.9152548, 1.1931061, -2.8520558, 3.5859926
2: -1.1156914, 2.7554140, -0.6048571, 1.2841841, -2.3998756, 3.3602712
3: -2.9853914, 3.3915496, -1.5523955, 1.5683012, -4.5536928, 4.9439449
4: -1.7583220, 3.5762279, -0.9047721, 1.7106876, -3.4690096, 4.4809999

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_A2_B1

### Relational analysis result of IS_A2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3861918, upper bound: 3.3861917
time: 0.44 seconds

## Relational analysis of IS_A2_B2_A2_A2_B2

### Relational analysis result of IS_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3861918, upper bound: 3.3861917
time: 0.44 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.34 seconds
IS_A1_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.34
Output dim: 0, lower bound: -3.3389717, upper bound: 3.2939083
IS_A1_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.34
Output dim: 0, lower bound: -3.3364705, upper bound: 3.2853451
IS_A1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.34
Output dim: 0, lower bound: -3.2643254, upper bound: 3.2713582
IS_A1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.34
Output dim: 0, lower bound: -3.2643254, upper bound: 3.3206361
IS_A1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.34
Output dim: 0, lower bound: -3.3359723, upper bound: 3.2908741
IS_A1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.34
Output dim: 0, lower bound: -3.3359908, upper bound: 3.2906396
IS_A1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.34
Output dim: 0, lower bound: -3.3433368, upper bound: 3.2984795
IS_A1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.34
Output dim: 0, lower bound: -3.3433368, upper bound: 3.2984795
IS_A1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.34
Output dim: 0, lower bound: -3.2832122, upper bound: 3.2859375
IS_A1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.34
Output dim: 0, lower bound: -3.2827588, upper bound: 3.2827588
IS_A1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.34
Output dim: 0, lower bound: -3.2736932, upper bound: 3.2620216
IS_A1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.34
Output dim: 0, lower bound: -3.2826483, upper bound: 3.2826483
IS_A1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.34
Output dim: 0, lower bound: -3.3592317, upper bound: 3.3072287
IS_A1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.34
Output dim: 0, lower bound: -3.3540666, upper bound: 3.2938665
IS_A1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.34
Output dim: 0, lower bound: -3.3349277, upper bound: 3.2700954
IS_A1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.34
Output dim: 0, lower bound: -3.3622409, upper bound: 3.3034641
IS_A2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.34
Output dim: 0, lower bound: -3.2680622, upper bound: 3.2869390
IS_A2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.34
Output dim: 0, lower bound: -3.2680622, upper bound: 3.2869390
IS_A2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.34
Output dim: 0, lower bound: -3.2825229, upper bound: 3.3441693
IS_A2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.34
Output dim: 0, lower bound: -3.2825229, upper bound: 3.3575497
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.34
Output dim: 0, lower bound: -3.2691295, upper bound: 3.2677982
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.34
Output dim: 0, lower bound: -3.3034837, upper bound: 3.3623153
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.34
Output dim: 0, lower bound: -3.2691295, upper bound: 3.2677982
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.34
Output dim: 0, lower bound: -3.3034837, upper bound: 3.3623154
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.34
Output dim: 0, lower bound: -3.3612467, upper bound: 3.3590332
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.34
Output dim: 0, lower bound: -3.3612467, upper bound: 3.3811898
IS_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.34
Output dim: 0, lower bound: -3.3742169, upper bound: 3.3732959
IS_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.34
Output dim: 0, lower bound: -3.3610161, upper bound: 3.3660411
IS_A2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.34
Output dim: 0, lower bound: -3.3743381, upper bound: 3.3657926
IS_A2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.34
Output dim: 0, lower bound: -3.3743381, upper bound: 3.3703173
IS_A2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.34
Output dim: 0, lower bound: -3.3861918, upper bound: 3.3861917
IS_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.34
Output dim: 0, lower bound: -3.3861918, upper bound: 3.3861917

## BFS IS instance: IS_A1_A1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.1479294, 0.0736306, -0.5053719, 0.7075273, -0.8554567, 0.5790025
1: -0.2056584, 0.1161741, -0.8093974, 1.0201974, -1.2258558, 0.9255713
2: -0.1902424, 0.1106335, -0.5398693, 1.1001620, -1.2904044, 0.6505027
3: -0.2053773, 0.1432274, -1.3487167, 1.3688184, -1.5741956, 1.4919441
4: -0.1201153, 0.1379478, -0.8305237, 1.4694343, -1.5895497, 0.9684715

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_A1_A1_A1

### Relational analysis result of IS_A1_A1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3358708, upper bound: 3.2884250
time: 0.43 seconds

## Relational analysis of IS_A1_A1_B1_A1_A1_A2

### Relational analysis result of IS_A1_A1_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3349206, upper bound: 3.2848771
time: 0.45 seconds

## BFS IS instance: IS_A1_A1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.1760671, 0.1537438, -0.5179793, 0.7358972, -0.9119642, 0.6717229
1: -0.2441616, 0.2217537, -0.8285949, 1.0636587, -1.3078203, 1.0503485
2: -0.2119544, 0.2359513, -0.5524985, 1.1419573, -1.3539118, 0.7884498
3: -0.3142368, 0.2612606, -1.3868341, 1.4223641, -1.7366009, 1.6480947
4: -0.1938589, 0.2861108, -0.8552358, 1.5194044, -1.7132633, 1.1413466

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 39

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_A1_B1_A1_A2_B1

### Relational analysis result of IS_A1_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3364705, upper bound: 3.2853353
time: 0.41 seconds

## Relational analysis of IS_A1_A1_B1_A1_A2_B2

### Relational analysis result of IS_A1_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3038641, upper bound: 3.2755314
time: 0.41 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.2088245, 0.2758393, -0.1844375, 0.1879934, -0.3968179, 0.4602767
1: -0.3110615, 0.3673706, -0.2575693, 0.2680588, -0.5791203, 0.6249399
2: -0.2585255, 0.3915462, -0.2257029, 0.2858377, -0.5443631, 0.6172491
3: -0.4219261, 0.4455681, -0.3429249, 0.3251451, -0.7470713, 0.7884930
4: -0.2464714, 0.4945604, -0.2149377, 0.3578165, -0.6042879, 0.7094982

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B1_B1

### Relational analysis result of IS_A1_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2643254, upper bound: 3.2713582
time: 0.42 seconds

## Relational analysis of IS_A1_A1_B1_A2_B1_B2

### Relational analysis result of IS_A1_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2643254, upper bound: 3.2713582
time: 0.50 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.2088245, 0.2758393, -0.6103234, 0.9568712, -1.1656957, 0.8861627
1: -0.3110615, 0.3673706, -0.9852375, 1.3554235, -1.6664850, 1.3526081
2: -0.2585255, 0.3915462, -0.6552639, 1.4612041, -1.7197295, 1.0468102
3: -0.4219261, 0.4455681, -1.7065320, 1.7653276, -2.1872537, 2.1521001
4: -0.2464714, 0.4945604, -0.9960408, 1.9181175, -2.1645889, 1.4906012

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2643254, upper bound: 3.3206361
time: 0.41 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B2

### Relational analysis result of IS_A1_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2643254, upper bound: 3.3206361
time: 0.45 seconds

## BFS IS instance: IS_A1_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.1030755, 0.0378904, -0.9136212, 1.5956707, -1.6987462, 0.9515116
1: -0.1123216, 0.0696629, -1.4485455, 2.2931776, -2.4054992, 1.5182084
2: -0.1210016, 0.0510990, -0.9755479, 2.3372707, -2.4582725, 1.0266469
3: -0.0784441, 0.0932648, -2.5513227, 2.9415061, -3.0199502, 2.6445875
4: -0.1141226, 0.0651078, -1.5737598, 3.0248685, -3.1389911, 1.6388675

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_A1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A1_B2_A1_A1_A1

### Relational analysis result of IS_A1_A1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3358708, upper bound: 3.2907357
time: 0.45 seconds

## Relational analysis of IS_A1_A1_B2_A1_A1_A2

### Relational analysis result of IS_A1_A1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3358708, upper bound: 3.2908741
time: 0.42 seconds

## BFS IS instance: IS_A1_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.1463613, 0.0593011, -0.9271833, 1.6220760, -1.7684374, 0.9864841
1: -0.2009166, 0.0985469, -1.4695764, 2.3377573, -2.5386739, 1.5681232
2: -0.1879099, 0.0829127, -0.9902581, 2.3730040, -2.5609138, 1.0731708
3: -0.1939110, 0.1251584, -2.5834830, 2.9980454, -3.1919565, 2.7086415
4: -0.1184179, 0.1058419, -1.6014781, 3.0695484, -3.1879661, 1.7073200

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A1_B2_A1_A2_A1

### Relational analysis result of IS_A1_A1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3333850, upper bound: 3.2801109
time: 0.46 seconds

## Relational analysis of IS_A1_A1_B2_A1_A2_A2

### Relational analysis result of IS_A1_A1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3333850, upper bound: 3.2906396
time: 0.43 seconds

## BFS IS instance: IS_A1_A1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.1493304, 0.0623187, -0.8719853, 1.5151944, -1.6645248, 0.9343039
1: -0.2054876, 0.0972578, -1.3874525, 2.1853328, -2.3908203, 1.4847102
2: -0.1923029, 0.0817966, -0.9315935, 2.2238195, -2.4161224, 1.0133899
3: -0.1952545, 0.1243617, -2.4344785, 2.8069587, -3.0022132, 2.5588403
4: -0.1209089, 0.1023947, -1.5067275, 2.8823516, -3.0032606, 1.6091220

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_A1_B2_A2_A1_B1

### Relational analysis result of IS_A1_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3433368, upper bound: 3.2983875
time: 0.41 seconds

## Relational analysis of IS_A1_A1_B2_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B2_A2_A1_B1

### Relational analysis result of IS_A1_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3433368, upper bound: 3.2984795
time: 0.43 seconds

## Relational analysis of IS_A1_A1_B2_A2_A1_B2

### Relational analysis result of IS_A1_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3433368, upper bound: 3.2984795
time: 0.43 seconds

## BFS IS instance: IS_A1_A1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.2020699, 0.2500587, -0.8719853, 1.5151944, -1.7172643, 1.1220440
1: -0.2900817, 0.3326781, -1.3874525, 2.1853328, -2.4754145, 1.7201304
2: -0.2458931, 0.3553001, -0.9315935, 2.2238195, -2.4697125, 1.2868936
3: -0.3762586, 0.3968206, -2.4344785, 2.8069587, -3.1832173, 2.8312991
4: -0.2341904, 0.4041437, -1.5067275, 2.8823516, -3.1165421, 1.9108711

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_A1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_A1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B2_A2_A2_B1

### Relational analysis result of IS_A1_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3433368, upper bound: 3.2984795
time: 0.40 seconds

## Relational analysis of IS_A1_A1_B2_A2_A2_B2

### Relational analysis result of IS_A1_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3433368, upper bound: 3.2984795
time: 0.46 seconds

## BFS IS instance: IS_A1_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1625811, 0.1084827, -0.1613177, 0.1075688, -0.2701499, 0.2698005
1: -0.2168661, 0.1631172, -0.2152069, 0.1666169, -0.3834831, 0.3783241
2: -0.1956260, 0.1684794, -0.1947221, 0.1664402, -0.3620661, 0.3632016
3: -0.2810408, 0.1911763, -0.2787543, 0.1955248, -0.4765655, 0.4699305
4: -0.1795351, 0.1994368, -0.1790864, 0.1967298, -0.3762649, 0.3785232

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_B1_B1_A1_A1

### Relational analysis result of IS_A1_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2768149, upper bound: 3.2575759
time: 0.40 seconds

## Relational analysis of IS_A1_A2_B1_B1_A1_A2

### Relational analysis result of IS_A1_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2894451, upper bound: 3.2831935
time: 0.40 seconds

## BFS IS instance: IS_A1_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.1805478, 0.1483264, -0.1874503, 0.1901035, -0.3706513, 0.3357767
1: -0.2465196, 0.2187737, -0.2646723, 0.2754298, -0.5219494, 0.4834460
2: -0.2190044, 0.2230366, -0.2303653, 0.2881553, -0.5071597, 0.4534020
3: -0.3176820, 0.2586381, -0.3487306, 0.3348741, -0.6525561, 0.6073687
4: -0.2011763, 0.2708065, -0.2225591, 0.3560021, -0.5571784, 0.4933656

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_B1_B1_A2_A1

### Relational analysis result of IS_A1_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2713582, upper bound: 3.2643254
time: 0.42 seconds

## Relational analysis of IS_A1_A2_B1_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_B1_B1_A2_B1

### Relational analysis result of IS_A1_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2800090, upper bound: 3.2778045
time: 0.44 seconds

## Relational analysis of IS_A1_A2_B1_B1_A2_B2

### Relational analysis result of IS_A1_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2860037, upper bound: 3.2793402
time: 0.44 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.1555604, 0.0857961, -0.1829544, 0.1576827, -0.3132431, 0.2687505
1: -0.2158905, 0.1343637, -0.2508442, 0.2308016, -0.4466920, 0.3852080
2: -0.2000806, 0.1217305, -0.2224463, 0.2378452, -0.4379258, 0.3441769
3: -0.2145249, 0.1663484, -0.3252409, 0.2737803, -0.4883052, 0.4915893
4: -0.1282436, 0.1528131, -0.2052960, 0.2909988, -0.4192424, 0.3581091

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A1_B1

### Relational analysis result of IS_A1_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2607417, upper bound: 3.2607417
time: 0.46 seconds

## Relational analysis of IS_A1_A2_B1_B2_A1_B2

### Relational analysis result of IS_A1_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2607417, upper bound: 3.2620216
time: 0.43 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.1538953, 0.0847333, -0.1621783, 0.0959122, -0.2498075, 0.2469117
1: -0.2140015, 0.1236971, -0.2132109, 0.1481941, -0.3621957, 0.3369080
2: -0.1986294, 0.1090919, -0.1948151, 0.1454921, -0.3441215, 0.3039070
3: -0.2086054, 0.1530888, -0.2747540, 0.1750011, -0.3836065, 0.4278429
4: -0.1261868, 0.1310454, -0.1778929, 0.1683002, -0.2944870, 0.3089383

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2620216, upper bound: 3.2736932
time: 0.44 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2620216, upper bound: 3.2826483
time: 0.43 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.1625811, 0.1084827, -0.4609864, 0.6584040, -0.8209851, 0.5694691
1: -0.2168661, 0.1631172, -0.7513965, 0.9260652, -1.1429313, 0.9145137
2: -0.1956260, 0.1684794, -0.5072581, 1.0205309, -1.2161570, 0.6757374
3: -0.2810408, 0.1911763, -1.2409196, 1.2295742, -1.5106150, 1.4320959
4: -0.1795351, 0.1994368, -0.7310875, 1.3581481, -1.5376832, 0.9305241

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_A2_B2_B1_A1_B1

### Relational analysis result of IS_A1_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3589416, upper bound: 3.3051587
time: 0.43 seconds

## Relational analysis of IS_A1_A2_B2_B1_A1_B2

### Relational analysis result of IS_A1_A2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3052061, upper bound: 3.2910252
time: 0.43 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.1805478, 0.1483264, -0.5722941, 0.8737229, -1.0542706, 0.7206205
1: -0.2465196, 0.2187737, -0.9319127, 1.2417749, -1.4882945, 1.1506864
2: -0.2190044, 0.2230366, -0.6209251, 1.3379942, -1.5569986, 0.8439617
3: -0.3176820, 0.2586381, -1.5913026, 1.6256982, -1.9433801, 1.8499404
4: -0.2011763, 0.2708065, -0.9329872, 1.7643747, -1.9655510, 1.2037936

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_B2_B1_A2_A1

### Relational analysis result of IS_A1_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3340069, upper bound: 3.2689630
time: 0.43 seconds

## Relational analysis of IS_A1_A2_B2_B1_A2_A2

### Relational analysis result of IS_A1_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3517838, upper bound: 3.2891232
time: 0.45 seconds

## BFS IS instance: IS_A1_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.1534767, 0.0731631, -0.7141318, 1.2211735, -1.3746502, 0.7872949
1: -0.2115669, 0.1165954, -1.1731652, 1.7171607, -1.9287276, 1.2897604
2: -0.1971034, 0.1021024, -0.7805954, 1.8135130, -2.0106163, 0.8826978
3: -0.2064490, 0.1461150, -2.0471113, 2.2136045, -2.4200535, 2.1932263
4: -0.1238867, 0.1306030, -1.2065284, 2.3437462, -2.4676328, 1.3371314

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_A2_B2_B2_A1_B1

### Relational analysis result of IS_A1_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3347516, upper bound: 3.2697512
time: 0.42 seconds

## Relational analysis of IS_A1_A2_B2_B2_A1_B2

### Relational analysis result of IS_A1_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2933439, upper bound: 3.2600638
time: 0.56 seconds

## BFS IS instance: IS_A1_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.1518041, 0.0714253, -0.5750132, 0.9110523, -1.0628564, 0.6464384
1: -0.2098211, 0.1045367, -0.9591896, 1.2886491, -1.4984702, 1.0637261
2: -0.1956123, 0.0906348, -0.6389191, 1.3820364, -1.5776488, 0.7295539
3: -0.2009172, 0.1313731, -1.6284152, 1.6890950, -1.8900121, 1.7597880
4: -0.1215191, 0.1104228, -0.9636544, 1.8103297, -1.9318488, 1.0740771

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_A2_B2_B2_A2_B1

### Relational analysis result of IS_A1_A2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3619314, upper bound: 3.3011127
time: 0.45 seconds

## Relational analysis of IS_A1_A2_B2_B2_A2_B2

### Relational analysis result of IS_A1_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2981604, upper bound: 3.2849770
time: 0.44 seconds

## BFS IS instance: IS_A2_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.3407221, 0.4752711, -0.1824498, 0.1752160, -0.5159381, 0.6577209
1: -0.5564522, 0.6531985, -0.2520449, 0.2516991, -0.8081513, 0.9052433
2: -0.3940860, 0.7215068, -0.2217253, 0.2701397, -0.6642257, 0.9432321
3: -0.8718258, 0.8605914, -0.3359327, 0.2997240, -1.1715498, 1.1965241
4: -0.5247028, 0.9477615, -0.2081054, 0.3357889, -0.8604916, 1.1558669

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A1_A1_B1_B1

### Relational analysis result of IS_A2_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2657479, upper bound: 3.2813415
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_A1_B1_B1

### Relational analysis result of IS_A2_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2603748, upper bound: 3.2789406
time: 0.44 seconds

## Relational analysis of IS_A2_B1_A1_A1_B1_B2

### Relational analysis result of IS_A2_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2680622, upper bound: 3.2869390
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.3407221, 0.4752711, -0.1992408, 0.2274800, -0.5682021, 0.6745119
1: -0.5564522, 0.6531985, -0.2820319, 0.3129402, -0.8693923, 0.9352304
2: -0.3940860, 0.7215068, -0.2451595, 0.3357430, -0.7298290, 0.9666663
3: -0.8718258, 0.8605914, -0.3751075, 0.3877977, -1.2596235, 1.2356989
4: -0.5247028, 0.9477615, -0.2453492, 0.4152069, -0.9399096, 1.1931107

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_A1_B2_B1

### Relational analysis result of IS_A2_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2680622, upper bound: 3.2869390
time: 0.44 seconds

## Relational analysis of IS_A2_B1_A1_A1_B2_B2

### Relational analysis result of IS_A2_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2680622, upper bound: 3.2869390
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.4084996, 0.5693522, -0.1824498, 0.1752160, -0.5837154, 0.7518020
1: -0.6703718, 0.7839134, -0.2520449, 0.2516991, -0.9220707, 1.0359583
2: -0.4531251, 0.8762918, -0.2217253, 0.2701397, -0.7232647, 1.0980170
3: -1.0841039, 1.0413346, -0.3359327, 0.2997240, -1.3838279, 1.3772674
4: -0.6257645, 1.1637021, -0.2081054, 0.3357889, -0.9615535, 1.3718076

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A1_A2_B1_B1

### Relational analysis result of IS_A2_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2772098, upper bound: 3.3239586
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_A2_B1_B1

### Relational analysis result of IS_A2_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2753743, upper bound: 3.3379815
time: 0.47 seconds

## Relational analysis of IS_A2_B1_A1_A2_B1_B2

### Relational analysis result of IS_A2_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2825229, upper bound: 3.3441692
time: 0.44 seconds

## BFS IS instance: IS_A2_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.4084996, 0.5693522, -0.1981089, 0.2274778, -0.6359773, 0.7674611
1: -0.6703718, 0.7839134, -0.2808432, 0.3129364, -0.9833078, 1.0647566
2: -0.4531251, 0.8762918, -0.2432533, 0.3357403, -0.7888653, 1.1195450
3: -1.0841039, 1.0413346, -0.3751075, 0.3858115, -1.4699154, 1.4164422
4: -0.6257645, 1.1637021, -0.2420689, 0.4152052, -1.0409697, 1.4057710

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_A2_B2_B1

### Relational analysis result of IS_A2_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2825229, upper bound: 3.3575497
time: 0.48 seconds

## Relational analysis of IS_A2_B1_A1_A2_B2_B2

### Relational analysis result of IS_A2_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2825229, upper bound: 3.3575497
time: 0.48 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.2885003, 0.3957064, -0.1847704, 0.1803702, -0.4688705, 0.5804768
1: -0.4617763, 0.5349889, -0.2591188, 0.2623837, -0.7241600, 0.7941077
2: -0.3472610, 0.6016396, -0.2265671, 0.2746382, -0.6218992, 0.8282068
3: -0.7067200, 0.6949047, -0.3410637, 0.3166220, -1.0233420, 1.0359683
4: -0.4197465, 0.7755752, -0.2161493, 0.3390406, -0.7587872, 0.9917244

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2702399, upper bound: 3.2677982
time: 0.44 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2702399, upper bound: 3.2677982
time: 0.45 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.3069124, 0.4287994, -0.1633160, 0.1176014, -0.4245138, 0.5921153
1: -0.5068933, 0.5922535, -0.2201020, 0.1792545, -0.6861478, 0.8123555
2: -0.3592525, 0.6426993, -0.1977011, 0.1814323, -0.5406848, 0.8404004
3: -0.7735788, 0.7753519, -0.2858234, 0.2103932, -0.9839720, 1.0611753
4: -0.4579649, 0.8279979, -0.1808549, 0.2160792, -0.6740441, 1.0088528

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2984795, upper bound: 3.3433367
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2984795, upper bound: 3.3433401
time: 0.46 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.2885003, 0.3957064, -0.1829544, 0.1576827, -0.4461831, 0.5786608
1: -0.4617763, 0.5349889, -0.2508442, 0.2308016, -0.6925779, 0.7858331
2: -0.3472610, 0.6016396, -0.2224463, 0.2378452, -0.5851063, 0.8240860
3: -0.7067200, 0.6949047, -0.3252409, 0.2737803, -0.9805003, 1.0201457
4: -0.4197465, 0.7755752, -0.2052960, 0.2909988, -0.7107453, 0.9808712

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2629600, upper bound: 3.2677982
time: 0.47 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2629600, upper bound: 3.2677982
time: 0.45 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.3069124, 0.4287994, -0.1621783, 0.0959122, -0.4028246, 0.5909777
1: -0.5068933, 0.5922535, -0.2132109, 0.1481941, -0.6550875, 0.8054644
2: -0.3592525, 0.6426993, -0.1948151, 0.1454921, -0.5047445, 0.8375144
3: -0.7735788, 0.7753519, -0.2747540, 0.1750011, -0.9485798, 1.0501060
4: -0.4579649, 0.8279979, -0.1778929, 0.1683002, -0.6262650, 1.0058908

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B2_A2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3011410, upper bound: 3.3619705
time: 0.46 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2853961, upper bound: 3.3010826
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.4108474, 0.5739069, -0.4892117, 0.7003186, -1.1111660, 1.0631186
1: -0.6647576, 0.7722933, -0.7909114, 0.9510068, -1.6157644, 1.5632048
2: -0.4539672, 0.8914004, -0.5302792, 1.0910958, -1.5450630, 1.4216795
3: -1.0951042, 1.0287758, -1.3417283, 1.2634159, -2.3585193, 2.3705034
4: -0.6342691, 1.2004530, -0.7672535, 1.4652178, -2.0994866, 1.9677063

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2830891, upper bound: 3.2756058
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B1_A1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3425915, upper bound: 3.3543105
time: 0.48 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3363146, upper bound: 3.3326538
time: 0.44 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.4633702, 0.6546778, -0.4892117, 0.7003186, -1.1636888, 1.1438895
1: -0.7499810, 0.9026212, -0.7909114, 0.9510068, -1.7009878, 1.6935326
2: -0.5000976, 1.0105180, -0.5302792, 1.0910958, -1.5911933, 1.5407971
3: -1.2505035, 1.1993570, -1.3417283, 1.2634159, -2.5139191, 2.5410850
4: -0.7164865, 1.3536885, -0.7672535, 1.4652178, -2.1817040, 2.1209414

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 37

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2830891, upper bound: 3.3576462
time: 0.45 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3401009, upper bound: 3.3336085
time: 0.50 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3363146, upper bound: 3.3326538
time: 0.47 seconds

## BFS IS instance: IS_A2_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.4754265, 0.6768659, -0.3075309, 0.4252267, -0.9006530, 0.9843968
1: -0.7717072, 0.9363350, -0.4999096, 0.5756644, -1.3473716, 1.4362445
2: -0.5144321, 1.0585876, -0.3536809, 0.6443481, -1.1587802, 1.4122684
3: -1.2923223, 1.2477293, -0.7712087, 0.7567580, -2.0490804, 2.0189381
4: -0.7464634, 1.4151686, -0.4553787, 0.8357342, -1.5821975, 1.8705473

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2830891, upper bound: 3.2761688
time: 0.46 seconds

## Relational analysis of IS_A2_B2_A1_B2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2830891, upper bound: 3.3732959
time: 0.50 seconds

## BFS IS instance: IS_A2_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.3089913, 0.4362090, -0.2457294, 0.3145940, -0.6235853, 0.6819384
1: -0.5045320, 0.6088690, -0.3677323, 0.4220365, -0.9265684, 0.9766012
2: -0.3574617, 0.6567069, -0.3022604, 0.4662319, -0.8236936, 0.9589673
3: -0.7698140, 0.7994115, -0.5331992, 0.5405312, -1.3103448, 1.3326107
4: -0.4737424, 0.8554390, -0.3159137, 0.5843010, -1.0580432, 1.1713526

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B2_B2_B1

### Relational analysis result of IS_A2_B2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3610161, upper bound: 3.3660411
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2_B2

### Relational analysis result of IS_A2_B2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3610161, upper bound: 3.3660412
time: 0.45 seconds

## BFS IS instance: IS_A2_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.7778159, 1.3267192, -0.6869586, 1.1129537, -1.8907689, 2.0136776
1: -1.2532948, 1.8665893, -1.1031027, 1.5574782, -2.8107727, 2.9696918
2: -0.8307147, 1.9636376, -0.7333291, 1.6772768, -2.5079906, 2.6969666
3: -2.2206659, 2.3894682, -1.9289154, 2.0171373, -4.2378035, 4.3183837
4: -1.2831259, 2.5623760, -1.1186684, 2.2024906, -3.4856162, 3.6810441

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A2_A1_B1_B1

### Relational analysis result of IS_A2_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3596432, upper bound: 3.3626779
time: 0.49 seconds

## Relational analysis of IS_A2_B2_A2_A1_B1_B2

### Relational analysis result of IS_A2_B2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3596432, upper bound: 3.3657925
time: 0.45 seconds

## BFS IS instance: IS_A2_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.7778159, 1.3267192, -1.1101861, 2.0660439, -2.8438594, 2.4369054
1: -1.2532948, 1.8665893, -1.7759900, 2.9424148, -4.1957092, 3.6425786
2: -0.8307147, 1.9636376, -1.1896495, 2.9707904, -3.8015049, 3.1532869
3: -2.2206659, 2.3894682, -3.1839290, 3.7166476, -5.9373131, 5.5733972
4: -1.2831259, 2.5623760, -1.8924820, 3.8328056, -5.1159315, 4.4548578

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A2_A1_B2_B1

### Relational analysis result of IS_A2_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3596432, upper bound: 3.3626779
time: 0.45 seconds

## Relational analysis of IS_A2_B2_A2_A1_B2_B2

### Relational analysis result of IS_A2_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3596432, upper bound: 3.3703173
time: 0.45 seconds

## BFS IS instance: IS_A2_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -1.0385535, 1.8943465, -0.4685251, 0.6649474, -1.7035009, 2.3628716
1: -1.6589499, 2.6707377, -0.7623996, 0.9295785, -2.5885284, 3.4331374
2: -1.1156914, 2.7554140, -0.5091282, 1.0344973, -2.1501887, 3.2645421
3: -2.9853914, 3.3915496, -1.2657418, 1.2390431, -4.2244344, 4.6572914
4: -1.7583220, 3.5762279, -0.7393889, 1.3829089, -3.1412311, 4.3156157

Time for backsubstitution: 2.33 seconds
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=3.5238394737243652
rel_dist={0: [-3.398203369636585, 3.398203369636585]}

## Binary Search with IS_dual Result
status: None
Maximum delta epsilon: None
execution time: 1131.63 seconds
