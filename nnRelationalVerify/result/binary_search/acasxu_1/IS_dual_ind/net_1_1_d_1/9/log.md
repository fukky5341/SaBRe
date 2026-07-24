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
execution time: IAR + LP analysis = 2.34 + 1.36 = 3.70 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -3.3982887, upper bound: 3.3982887


# Binary Search by BASE starts (time budget: 1196.30 seconds, max iter: 100)

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
Binary search time: 67.75 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 1128.55 seconds

## Binary search (step 0) starts
Candidate diff: 0.2500000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3784408, upper bound: 3.3327265
time: 0.40 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3945339, upper bound: 3.3945338
time: 0.42 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.00 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.00
Output dim: 0, lower bound: -3.3784408, upper bound: 3.3327265
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.00
Output dim: 0, lower bound: -3.3945339, upper bound: 3.3945338

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.2895984, 0.3943615, -1.2466472, 2.2771921, -2.5667903, 1.6410087
1: -0.4578271, 0.5641531, -1.9637374, 3.1950235, -3.6528506, 2.5278900
2: -0.3370600, 0.5814233, -1.3604455, 3.2710245, -3.6080840, 1.9418688
3: -0.6758766, 0.7399411, -3.4595599, 4.0564485, -4.7323251, 4.1995006
4: -0.4528955, 0.7513722, -2.1785955, 4.2066536, -4.6595483, 2.9299674

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3169978, upper bound: 3.3169978
time: 0.52 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3169978, upper bound: 3.3327265
time: 0.47 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -1.0333478, 1.8749698, -1.2193484, 2.2206030, -3.2539508, 3.0943177
1: -1.6493901, 2.6237774, -1.9220543, 3.1182246, -4.7676139, 4.5458307
2: -1.1160958, 2.7230368, -1.3294032, 3.1937323, -4.3098278, 4.0524397
3: -2.9354391, 3.3341267, -3.3847332, 3.9624071, -6.8978462, 6.7188597
4: -1.7549448, 3.5216486, -2.1279650, 4.1108456, -5.8657894, 5.6496134

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3327265, upper bound: 3.3784408
time: 0.40 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3327265, upper bound: 3.3945339
time: 0.42 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.22 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.22
Output dim: 0, lower bound: -3.3169978, upper bound: 3.3169978
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.22
Output dim: 0, lower bound: -3.3169978, upper bound: 3.3327265
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.22
Output dim: 0, lower bound: -3.3327265, upper bound: 3.3784408
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.22
Output dim: 0, lower bound: -3.3327265, upper bound: 3.3945339

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.2895984, 0.3943615, -0.2895984, 0.3943615, -0.6839598, 0.6839598
1: -0.4578271, 0.5641531, -0.4578271, 0.5641531, -1.0219798, 1.0219798
2: -0.3370600, 0.5814233, -0.3370600, 0.5814233, -0.9184830, 0.9184831
3: -0.6758766, 0.7399411, -0.6758766, 0.7399411, -1.4158176, 1.4158175
4: -0.4528955, 0.7513722, -0.4528955, 0.7513722, -1.2042676, 1.2042676

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3169505, upper bound: 3.3169978
time: 0.38 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3169978, upper bound: 3.3169978
time: 0.40 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.2895984, 0.3943615, -1.0333478, 1.8749698, -2.1645675, 1.4277093
1: -0.4578271, 0.5641531, -1.6493901, 2.6237774, -3.0816045, 2.2135432
2: -0.3370600, 0.5814233, -1.1160958, 2.7230368, -3.0600965, 1.6975191
3: -0.6758766, 0.7399411, -2.9354391, 3.3341267, -4.0100031, 3.6753800
4: -0.4528955, 0.7513722, -1.7549448, 3.5216486, -3.9745436, 2.5063167

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3169505, upper bound: 3.3327265
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3169978, upper bound: 3.3327265
time: 0.41 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -1.0333478, 1.8749698, -0.2895984, 0.3943615, -1.4277093, 2.1645677
1: -1.6493901, 2.6237774, -0.4578271, 0.5641531, -2.2135432, 3.0816045
2: -1.1160958, 2.7230368, -0.3370600, 0.5814233, -1.6975191, 3.0600965
3: -2.9354391, 3.3341267, -0.6758766, 0.7399411, -3.6753798, 4.0100031
4: -1.7549448, 3.5216486, -0.4528955, 0.7513722, -2.5063167, 3.9745436

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3213076, upper bound: 3.3740498
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3067627, upper bound: 3.3150157
time: 0.40 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -1.0333478, 1.8749698, -1.0333478, 1.8749698, -2.9083173, 2.9083176
1: -1.6493901, 2.6237774, -1.6493901, 2.6237774, -4.2731667, 4.2731667
2: -1.1160958, 2.7230368, -1.1160958, 2.7230368, -3.8391325, 3.8391325
3: -2.9354391, 3.3341267, -2.9354391, 3.3341267, -6.2695656, 6.2695656
4: -1.7549448, 3.5216486, -1.7549448, 3.5216486, -5.2765932, 5.2765932

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3213076, upper bound: 3.3767914
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3067627, upper bound: 3.3205784
time: 0.44 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.10 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.10
Output dim: 0, lower bound: -3.3169505, upper bound: 3.3169978
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.10
Output dim: 0, lower bound: -3.3169978, upper bound: 3.3169978
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.10
Output dim: 0, lower bound: -3.3169505, upper bound: 3.3327265
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.10
Output dim: 0, lower bound: -3.3169978, upper bound: 3.3327265
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.10
Output dim: 0, lower bound: -3.3213076, upper bound: 3.3740498
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.10
Output dim: 0, lower bound: -3.3067627, upper bound: 3.3150157
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.10
Output dim: 0, lower bound: -3.3213076, upper bound: 3.3767914
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.10
Output dim: 0, lower bound: -3.3067627, upper bound: 3.3205784

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1912085, 0.2109509, -0.2895984, 0.3943615, -0.5855700, 0.5005493
1: -0.2706243, 0.2982620, -0.4578271, 0.5641531, -0.8347774, 0.7560890
2: -0.2345207, 0.3197280, -0.3370600, 0.5814233, -0.8159440, 0.6567879
3: -0.3691734, 0.3672814, -0.6758766, 0.7399411, -1.1091145, 1.0431577
4: -0.2326370, 0.4024982, -0.4528955, 0.7513722, -0.9840093, 0.8553935

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3169505, upper bound: 3.3169505
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3169505, upper bound: 3.3169978
time: 0.39 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.4519231, 0.6203508, -0.2863977, 0.3894633, -0.8413863, 0.9067485
1: -0.7154577, 0.8934209, -0.4525831, 0.5572431, -1.2727008, 1.3460039
2: -0.4850966, 0.9528847, -0.3339674, 0.5737467, -1.0588431, 1.2868520
3: -1.1693269, 1.1907833, -0.6662335, 0.7304243, -1.8997512, 1.8570168
4: -0.7152326, 1.2680461, -0.4470947, 0.7407788, -1.4560113, 1.7151407

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3169978, upper bound: 3.3169505
time: 0.38 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3169978, upper bound: 3.3169978
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.1912085, 0.2109509, -1.0333478, 1.8749698, -2.0661783, 1.2442987
1: -0.2706243, 0.2982620, -1.6493901, 2.6237774, -2.8944016, 1.9476519
2: -0.2345207, 0.3197280, -1.1160958, 2.7230368, -2.9575574, 1.4358237
3: -0.3691734, 0.3672814, -2.9354391, 3.3341267, -3.7033002, 3.3027203
4: -0.2326370, 0.4024982, -1.7549448, 3.5216486, -3.7542858, 2.1574428

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3366947, upper bound: 3.2076429
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2827253, upper bound: 3.1944766
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.4519231, 0.6203508, -1.0238039, 1.8541665, -2.3060896, 1.6441548
1: -0.7154577, 0.8934209, -1.6348212, 2.5962243, -3.3116817, 2.5282419
2: -0.4850966, 0.9528847, -1.1053314, 2.6938963, -3.1789927, 2.0582159
3: -1.1693269, 1.1907833, -2.9082785, 3.3003573, -4.4696841, 4.0990601
4: -0.7152326, 1.2680461, -1.7376633, 3.4854763, -4.2007084, 3.0057094

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3740498, upper bound: 3.3213076
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3150157, upper bound: 3.3067627
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.3753012, 0.5288189, -0.2895984, 0.3943615, -0.7696627, 0.8184172
1: -0.6084703, 0.7558654, -0.4578271, 0.5641531, -1.1726230, 1.2136924
2: -0.4198778, 0.7994800, -0.3370600, 0.5814233, -1.0013011, 1.1365399
3: -0.9523419, 0.9947283, -0.6758766, 0.7399411, -1.6922829, 1.6706049
4: -0.5820951, 1.0479591, -0.4528955, 0.7513722, -1.3334674, 1.5008545

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2076429, upper bound: 3.3366947
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3213076, upper bound: 3.3740498
time: 0.45 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.6188753, 0.9653776, -0.2895984, 0.3943615, -1.0132368, 1.2549759
1: -1.0178568, 1.3381594, -0.4578271, 0.5641531, -1.5820097, 1.7959864
2: -0.6695868, 1.4706283, -0.3370600, 0.5814233, -1.2510102, 1.8076882
3: -1.7465825, 1.7551888, -0.6758766, 0.7399411, -2.4865236, 2.4310656
4: -1.0122970, 1.9336305, -0.4528955, 0.7513722, -1.7636693, 2.3865259

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1944766, upper bound: 3.2827253
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3067627, upper bound: 3.3150158
time: 0.40 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.3753012, 0.5288189, -1.0333478, 1.8749698, -2.2502706, 1.5621667
1: -0.6084703, 0.7558654, -1.6493901, 2.6237774, -3.2322476, 2.4052553
2: -0.4198778, 0.7994800, -1.1160958, 2.7230368, -3.1429145, 1.9155757
3: -0.9523419, 0.9947283, -2.9354391, 3.3341267, -4.2864685, 3.9301674
4: -0.5820951, 1.0479591, -1.7549448, 3.5216486, -4.1037436, 2.8029034

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3205783, upper bound: 3.3205783
time: 0.44 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3205783, upper bound: 3.3205783
time: 0.52 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.6188753, 0.9653776, -1.0333478, 1.8749698, -2.4938445, 1.9987254
1: -1.0178568, 1.3381594, -1.6493901, 2.6237774, -3.6416342, 2.9875493
2: -0.6695868, 1.4706283, -1.1160958, 2.7230368, -3.3926237, 2.5867240
3: -1.7465825, 1.7551888, -2.9354391, 3.3341267, -5.0807095, 4.6906271
4: -1.0122970, 1.9336305, -1.7549448, 3.5216486, -4.5339456, 3.6885748

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3205783, upper bound: 3.3205783
time: 0.43 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3205783, upper bound: 3.3205783
time: 0.44 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.14 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.14
Output dim: 0, lower bound: -3.3169505, upper bound: 3.3169505
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.14
Output dim: 0, lower bound: -3.3169505, upper bound: 3.3169978
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.14
Output dim: 0, lower bound: -3.3169978, upper bound: 3.3169505
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.14
Output dim: 0, lower bound: -3.3169978, upper bound: 3.3169978
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.14
Output dim: 0, lower bound: -3.3366947, upper bound: 3.2076429
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.14
Output dim: 0, lower bound: -3.2827253, upper bound: 3.1944766
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.14
Output dim: 0, lower bound: -3.3740498, upper bound: 3.3213076
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.14
Output dim: 0, lower bound: -3.3150157, upper bound: 3.3067627
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.14
Output dim: 0, lower bound: -3.2076429, upper bound: 3.3366947
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.14
Output dim: 0, lower bound: -3.3213076, upper bound: 3.3740498
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.14
Output dim: 0, lower bound: -3.1944766, upper bound: 3.2827253
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.14
Output dim: 0, lower bound: -3.3067627, upper bound: 3.3150158
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.14
Output dim: 0, lower bound: -3.3205783, upper bound: 3.3205783
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.14
Output dim: 0, lower bound: -3.3205783, upper bound: 3.3205783
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.14
Output dim: 0, lower bound: -3.3205783, upper bound: 3.3205783
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.14
Output dim: 0, lower bound: -3.3205783, upper bound: 3.3205783

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1912085, 0.2109509, -0.1912085, 0.2109509, -0.4021594, 0.4021594
1: -0.2706243, 0.2982620, -0.2706243, 0.2982620, -0.5688863, 0.5688863
2: -0.2345207, 0.3197280, -0.2345207, 0.3197280, -0.5542487, 0.5542487
3: -0.3691734, 0.3672814, -0.3691734, 0.3672814, -0.7364548, 0.7364548
4: -0.2326370, 0.4024982, -0.2326370, 0.4024982, -0.6351352, 0.6351352

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1898258, upper bound: 3.2772939
time: 0.38 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1566236, upper bound: 3.1566236
time: 0.36 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1912085, 0.2109509, -0.4519231, 0.6203508, -0.8115593, 0.6628739
1: -0.2706243, 0.2982620, -0.7154577, 0.8934209, -1.1640452, 1.0137197
2: -0.2345207, 0.3197280, -0.4850966, 0.9528847, -1.1874055, 0.8048245
3: -0.3691734, 0.3672814, -1.1693269, 1.1907833, -1.5599567, 1.5366082
4: -0.2326370, 0.4024982, -0.7152326, 1.2680461, -1.5006832, 1.1177306

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1898258, upper bound: 3.3148614
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1566236, upper bound: 3.1889140
time: 0.38 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.4519231, 0.6203508, -0.1912085, 0.2109509, -0.6628739, 0.8115593
1: -0.7154577, 0.8934209, -0.2706243, 0.2982620, -1.0137197, 1.1640452
2: -0.4850966, 0.9528847, -0.2345207, 0.3197280, -0.8048245, 1.1874055
3: -1.1693269, 1.1907833, -0.3691734, 0.3672814, -1.5366082, 1.5599567
4: -0.7152326, 1.2680461, -0.2326370, 0.4024982, -1.1177306, 1.5006832

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3087832, upper bound: 3.3054752
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3098656, upper bound: 3.3098627
time: 0.41 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.4519231, 0.6203508, -0.4519231, 0.6203508, -1.0722739, 1.0722739
1: -0.7154577, 0.8934209, -0.7154577, 0.8934209, -1.6088786, 1.6088786
2: -0.4850966, 0.9528847, -0.4850966, 0.9528847, -1.4379812, 1.4379812
3: -1.1693269, 1.1907833, -1.1693269, 1.1907833, -2.3601103, 2.3601103
4: -0.7152326, 1.2680461, -0.7152326, 1.2680461, -1.9832784, 1.9832784

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3087832, upper bound: 3.3054781
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3098656, upper bound: 3.3098627
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.1912085, 0.2109509, -0.3753012, 0.5288189, -0.7200274, 0.5862522
1: -0.2706243, 0.2982620, -0.6084703, 0.7558654, -1.0264896, 0.9067323
2: -0.2345207, 0.3197280, -0.4198778, 0.7994800, -1.0340008, 0.7396057
3: -0.3691734, 0.3672814, -0.9523419, 0.9947283, -1.3639017, 1.3196230
4: -0.2326370, 0.4024982, -0.5820951, 1.0479591, -1.2805961, 0.9845932

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3127877, upper bound: 3.1940192
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2827253, upper bound: 3.1944766
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2827253, upper bound: 3.1944766
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.1912085, 0.2109509, -0.6188753, 0.9653776, -1.1565861, 0.8298262
1: -0.2706243, 0.2982620, -1.0178568, 1.3381594, -1.6087837, 1.3161187
2: -0.2345207, 0.3197280, -0.6695868, 1.4706283, -1.7051491, 0.9893147
3: -0.3691734, 0.3672814, -1.7465825, 1.7551888, -2.1243622, 2.1138637
4: -0.2326370, 0.4024982, -1.0122970, 1.9336305, -2.1662674, 1.4147952

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2827253, upper bound: 3.1944766
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2827253, upper bound: 3.1944766
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.4519231, 0.6203508, -0.3722968, 0.5244855, -0.9764084, 0.9926476
1: -0.7154577, 0.8934209, -0.6036323, 0.7497613, -1.4652190, 1.4970528
2: -0.4850966, 0.9528847, -0.4169547, 0.7923486, -1.2774448, 1.3698393
3: -1.1693269, 1.1907833, -0.9431840, 0.9862554, -2.1555824, 2.1339672
4: -0.7152326, 1.2680461, -0.5769094, 1.0379164, -1.7531490, 1.8449554

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3666185, upper bound: 3.3127592
time: 0.43 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3150157, upper bound: 3.3067627
time: 0.43 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3150157, upper bound: 3.3067627
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.4519231, 0.6203508, -0.6115561, 0.9495890, -1.4015120, 1.2319069
1: -0.7154577, 0.8934209, -1.0066755, 1.3170630, -2.0325208, 1.9000961
2: -0.4850966, 0.9528847, -0.6623885, 1.4479090, -1.9330053, 1.6152731
3: -1.1693269, 1.1907833, -1.7245955, 1.7292082, -2.8985353, 2.9153788
4: -0.7152326, 1.2680461, -1.0000154, 1.9054065, -2.6206388, 2.2680614

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3074373, upper bound: 3.2972614
time: 0.39 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3150158, upper bound: 3.3067627
time: 0.44 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3150158, upper bound: 3.3067627
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.3753012, 0.5288189, -0.1912085, 0.2109509, -0.5862522, 0.7200274
1: -0.6084703, 0.7558654, -0.2706243, 0.2982620, -0.9067323, 1.0264896
2: -0.4198778, 0.7994800, -0.2345207, 0.3197280, -0.7396057, 1.0340008
3: -0.9523419, 0.9947283, -0.3691734, 0.3672814, -1.3196230, 1.3639017
4: -0.5820951, 1.0479591, -0.2326370, 0.4024982, -0.9845930, 1.2805961

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2076429, upper bound: 3.3366947
time: 0.44 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2076429, upper bound: 3.3366947
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.3722968, 0.5244855, -0.4519231, 0.6203508, -0.9926476, 0.9764084
1: -0.6036323, 0.7497613, -0.7154577, 0.8934209, -1.4970528, 1.4652190
2: -0.4169547, 0.7923486, -0.4850966, 0.9528847, -1.3698393, 1.2774448
3: -0.9431840, 0.9862554, -1.1693269, 1.1907833, -2.1339672, 2.1555824
4: -0.5769094, 1.0379164, -0.7152326, 1.2680461, -1.8449553, 1.7531490

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3212949, upper bound: 3.3739734
time: 0.45 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3212949, upper bound: 3.3740498
time: 0.44 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.6188753, 0.9653776, -0.1912085, 0.2109509, -0.8298262, 1.1565861
1: -1.0178568, 1.3381594, -0.2706243, 0.2982620, -1.3161187, 1.6087837
2: -0.6695868, 1.4706283, -0.2345207, 0.3197280, -0.9893148, 1.7051491
3: -1.7465825, 1.7551888, -0.3691734, 0.3672814, -2.1138637, 2.1243622
4: -1.0122970, 1.9336305, -0.2326370, 0.4024982, -1.4147952, 2.1662674

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1944766, upper bound: 3.2827253
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1944632, upper bound: 3.2826990
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.6115561, 0.9495890, -0.4519231, 0.6203508, -1.2319069, 1.4015120
1: -1.0066755, 1.3170630, -0.7154577, 0.8934209, -1.9000958, 2.0325208
2: -0.6623885, 1.4479090, -0.4850966, 0.9528847, -1.6152732, 1.9330051
3: -1.7245955, 1.7292082, -1.1693269, 1.1907833, -2.9153788, 2.8985353
4: -1.0000154, 1.9054065, -0.7152326, 1.2680461, -2.2680614, 2.6206388

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0831486, upper bound: 3.2327070
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3038922, upper bound: 3.3150157
time: 0.46 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3067627, upper bound: 3.3149895
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.3753012, 0.5288189, -0.3753012, 0.5288189, -0.9041201, 0.9041200
1: -0.6084703, 0.7558654, -0.6084703, 0.7558654, -1.3643357, 1.3643357
2: -0.4198778, 0.7994800, -0.4198778, 0.7994800, -1.2193577, 1.2193577
3: -0.9523419, 0.9947283, -0.9523419, 0.9947283, -1.9470696, 1.9470699
4: -0.5820951, 1.0479591, -0.5820951, 1.0479591, -1.6300540, 1.6300540

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3338167, upper bound: 3.3767067
time: 0.43 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3338116, upper bound: 3.3767913
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.3753012, 0.5288189, -0.6188753, 0.9653776, -1.3406787, 1.1476941
1: -0.6084703, 0.7558654, -1.0178568, 1.3381594, -1.9466296, 1.7737222
2: -0.4198778, 0.7994800, -0.6695868, 1.4706283, -1.8905060, 1.4690667
3: -0.9523419, 0.9947283, -1.7465825, 1.7551888, -2.7075305, 2.7413108
4: -0.5820951, 1.0479591, -1.0122970, 1.9336305, -2.5157256, 2.0602558

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3338167, upper bound: 3.3767067
time: 0.45 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3338116, upper bound: 3.3767913
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.6188753, 0.9653776, -0.3753012, 0.5288189, -1.1476941, 1.3406787
1: -1.0178568, 1.3381594, -0.6084703, 0.7558654, -1.7737221, 1.9466296
2: -0.6695868, 1.4706283, -0.4198778, 0.7994800, -1.4690667, 1.8905060
3: -1.7465825, 1.7551888, -0.9523419, 0.9947283, -2.7413106, 2.7075305
4: -1.0122970, 1.9336305, -0.5820951, 1.0479591, -2.0602555, 2.5157256

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3134799, upper bound: 3.3134291
time: 0.47 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3135253, upper bound: 3.3135253
time: 0.46 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.6188753, 0.9653776, -0.6188753, 0.9653776, -1.5842528, 1.5842528
1: -1.0178568, 1.3381594, -1.0178568, 1.3381594, -2.3560159, 2.3560159
2: -0.6695868, 1.4706283, -0.6695868, 1.4706283, -2.1402152, 2.1402152
3: -1.7465825, 1.7551888, -1.7465825, 1.7551888, -3.5017715, 3.5017715
4: -1.0122970, 1.9336305, -1.0122970, 1.9336305, -2.9459276, 2.9459276

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3134799, upper bound: 3.3134291
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3135253, upper bound: 3.3135253
time: 0.46 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.57 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 0, lower bound: -3.1898258, upper bound: 3.2772939
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 0, lower bound: -3.1566236, upper bound: 3.1566236
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 0, lower bound: -3.1898258, upper bound: 3.3148614
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 0, lower bound: -3.1566236, upper bound: 3.1889140
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 0, lower bound: -3.3087832, upper bound: 3.3054752
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 0, lower bound: -3.3098656, upper bound: 3.3098627
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 0, lower bound: -3.3087832, upper bound: 3.3054781
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 0, lower bound: -3.3098656, upper bound: 3.3098627
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 0, lower bound: -3.2827253, upper bound: 3.1944766
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 0, lower bound: -3.2827253, upper bound: 3.1944766
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 0, lower bound: -3.2827253, upper bound: 3.1944766
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 0, lower bound: -3.2827253, upper bound: 3.1944766
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 0, lower bound: -3.3150157, upper bound: 3.3067627
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 0, lower bound: -3.3150157, upper bound: 3.3067627
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 0, lower bound: -3.3150158, upper bound: 3.3067627
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 0, lower bound: -3.3150158, upper bound: 3.3067627
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 0, lower bound: -3.2076429, upper bound: 3.3366947
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 0, lower bound: -3.2076429, upper bound: 3.3366947
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 0, lower bound: -3.3212949, upper bound: 3.3739734
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 0, lower bound: -3.3212949, upper bound: 3.3740498
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 0, lower bound: -3.1944766, upper bound: 3.2827253
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 0, lower bound: -3.1944632, upper bound: 3.2826990
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 0, lower bound: -3.3038922, upper bound: 3.3150157
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 0, lower bound: -3.3067627, upper bound: 3.3149895
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 0, lower bound: -3.3338167, upper bound: 3.3767067
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 0, lower bound: -3.3338116, upper bound: 3.3767913
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 0, lower bound: -3.3338167, upper bound: 3.3767067
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 0, lower bound: -3.3338116, upper bound: 3.3767913
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 0, lower bound: -3.3134799, upper bound: 3.3134291
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 0, lower bound: -3.3135253, upper bound: 3.3135253
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 0, lower bound: -3.3134799, upper bound: 3.3134291
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.57
Output dim: 0, lower bound: -3.3135253, upper bound: 3.3135253

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0676768, 0.0203616, -0.1912085, 0.2109509, -0.2786278, 0.2115700
1: -0.0773866, 0.0391565, -0.2706243, 0.2982620, -0.3756486, 0.3097808
2: -0.0644345, 0.0282484, -0.2345207, 0.3197280, -0.3841625, 0.2627692
3: -0.0712000, 0.0471544, -0.3691734, 0.3672814, -0.4384814, 0.4163279
4: -0.0532835, 0.0354563, -0.2326370, 0.4024982, -0.4557817, 0.2680933

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1763442, upper bound: 3.2521502
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1566236, upper bound: 3.1566236
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1566236, upper bound: 3.1566236
time: 0.37 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.1649871, 0.1174907, -0.1912085, 0.2109509, -0.3759381, 0.3086992
1: -0.2215959, 0.1765458, -0.2706243, 0.2982620, -0.5198579, 0.4471700
2: -0.1996021, 0.1877475, -0.2345207, 0.3197280, -0.5193301, 0.4222682
3: -0.2914082, 0.2083298, -0.3691734, 0.3672814, -0.6586896, 0.5775033
4: -0.1816220, 0.2294925, -0.2326370, 0.4024982, -0.5841202, 0.4621295

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1566236, upper bound: 3.1566236
time: 0.43 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1566236, upper bound: 3.1566236
time: 0.44 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0676768, 0.0203616, -0.4519231, 0.6203508, -0.6880276, 0.4722845
1: -0.0773866, 0.0391565, -0.7154577, 0.8934209, -0.9708076, 0.7546142
2: -0.0644345, 0.0282484, -0.4850966, 0.9528847, -1.0173192, 0.5133449
3: -0.0712000, 0.0471544, -1.1693269, 1.1907833, -1.2619833, 1.2164813
4: -0.0532835, 0.0354563, -0.7152326, 1.2680461, -1.3213297, 0.7506888

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2974989, upper bound: 3.3071032
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2674166, upper bound: 3.1889140
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2674166, upper bound: 3.1889140
time: 0.41 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.1649871, 0.1174907, -0.4519231, 0.6203508, -0.7853379, 0.5694137
1: -0.2215959, 0.1765458, -0.7154577, 0.8934209, -1.1150168, 0.8920035
2: -0.1996021, 0.1877475, -0.4850966, 0.9528847, -1.1524868, 0.6728441
3: -0.2914082, 0.2083298, -1.1693269, 1.1907833, -1.4821914, 1.3776567
4: -0.1816220, 0.2294925, -0.7152326, 1.2680461, -1.4496682, 0.9447250

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2674166, upper bound: 3.1889140
time: 0.44 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2674166, upper bound: 3.1889140
time: 0.39 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.2559387, 0.3415756, -0.1912085, 0.2109509, -0.4668896, 0.5327840
1: -0.3946697, 0.4863652, -0.2706243, 0.2982620, -0.6929317, 0.7569895
2: -0.2981569, 0.5062003, -0.2345207, 0.3197280, -0.6178848, 0.7407210
3: -0.5855743, 0.6284050, -0.3691734, 0.3672814, -0.9528557, 0.9975784
4: -0.3760676, 0.6415557, -0.2326370, 0.4024982, -0.7785658, 0.8741927

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3071032, upper bound: 3.2974989
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3056196, upper bound: 3.3043631
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3087832, upper bound: 3.3054752
time: 0.41 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.4087062, 0.5647650, -0.1866335, 0.1983020, -0.6070081, 0.7513984
1: -0.6539204, 0.7966581, -0.2622981, 0.2809657, -0.9348860, 1.0589563
2: -0.4457314, 0.8629469, -0.2285777, 0.3013574, -0.7470888, 1.0915245
3: -1.0570911, 1.0608249, -0.3510454, 0.3435058, -1.4005969, 1.4118702
4: -0.6360155, 1.1408229, -0.2222428, 0.3786544, -1.0146699, 1.3630657

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3067020, upper bound: 3.3087506
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3098656, upper bound: 3.3098627
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.2559387, 0.3415756, -0.4519231, 0.6203508, -0.8762895, 0.7934986
1: -0.3946697, 0.4863652, -0.7154577, 0.8934209, -1.2880906, 1.2018229
2: -0.2981569, 0.5062003, -0.4850966, 0.9528847, -1.2510415, 0.9912968
3: -0.5855743, 0.6284050, -1.1693269, 1.1907833, -1.7763577, 1.7977316
4: -0.3760676, 0.6415557, -0.7152326, 1.2680461, -1.6441138, 1.3567882

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3043957, upper bound: 3.3043813
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3043957, upper bound: 3.3054781
time: 0.45 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.4087062, 0.5647650, -0.4365533, 0.5996151, -1.0083212, 1.0013176
1: -0.6539204, 0.7966581, -0.6914814, 0.8636861, -1.5176065, 1.4881394
2: -0.4457314, 0.8629469, -0.4703101, 0.9182796, -1.3640110, 1.3332568
3: -1.0570911, 1.0608249, -1.1243591, 1.1498823, -2.2069726, 2.1851835
4: -0.6360155, 1.1408229, -0.6897490, 1.2193394, -1.8553545, 1.8305712

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3053682, upper bound: 3.3086207
time: 0.43 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3053682, upper bound: 3.3098627
time: 0.43 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0676768, 0.0203616, -0.3753012, 0.5288189, -0.5964957, 0.3956628
1: -0.0773866, 0.0391565, -0.6084703, 0.7558654, -0.8332520, 0.6476268
2: -0.0644345, 0.0282484, -0.4198778, 0.7994800, -0.8639145, 0.4481262
3: -0.0712000, 0.0471544, -0.9523419, 0.9947283, -1.0659282, 0.9994963
4: -0.0532835, 0.0354563, -0.5820951, 1.0479591, -1.1012427, 0.6175514

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3366947, upper bound: 3.2076429
time: 0.43 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3366947, upper bound: 3.2076429
time: 0.46 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.1649871, 0.1174907, -0.3753012, 0.5288189, -0.6938061, 0.4927919
1: -0.2215959, 0.1765458, -0.6084703, 0.7558654, -0.9774612, 0.7850159
2: -0.1996021, 0.1877475, -0.4198778, 0.7994800, -0.9990821, 0.6076252
3: -0.2914082, 0.2083298, -0.9523419, 0.9947283, -1.2861365, 1.1606716
4: -0.1816220, 0.2294925, -0.5820951, 1.0479591, -1.2295811, 0.8115876

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3366947, upper bound: 3.2076429
time: 0.46 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3366947, upper bound: 3.2076429
time: 0.44 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0676768, 0.0203616, -0.6188753, 0.9653776, -1.0330544, 0.6392368
1: -0.0773866, 0.0391565, -1.0178568, 1.3381594, -1.4155461, 1.0570132
2: -0.0644345, 0.0282484, -0.6695868, 1.4706283, -1.5350628, 0.6978353
3: -0.0712000, 0.0471544, -1.7465825, 1.7551888, -1.8263888, 1.7937369
4: -0.0532835, 0.0354563, -1.0122970, 1.9336305, -1.9869140, 1.0477532

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2827253, upper bound: 3.1944766
time: 0.50 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2826990, upper bound: 3.1944632
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.1649871, 0.1174907, -0.6188753, 0.9653776, -1.1303647, 0.7363660
1: -0.2215959, 0.1765458, -1.0178568, 1.3381594, -1.5597553, 1.1944026
2: -0.1996021, 0.1877475, -0.6695868, 1.4706283, -1.6702304, 0.8573343
3: -0.2914082, 0.2083298, -1.7465825, 1.7551888, -2.0465970, 1.9549123
4: -0.1816220, 0.2294925, -1.0122970, 1.9336305, -2.1152525, 1.2417895

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2827253, upper bound: 3.1944766
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2826990, upper bound: 3.1944632
time: 0.42 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.1850611, 0.2096545, -0.3722968, 0.5244855, -0.7095466, 0.5819513
1: -0.2660490, 0.2913542, -0.6036323, 0.7497613, -1.0158104, 0.8949865
2: -0.2290718, 0.3008571, -0.4169547, 0.7923486, -1.0214204, 0.7178117
3: -0.3428191, 0.3601812, -0.9431840, 0.9862554, -1.3290745, 1.3033652
4: -0.2267372, 0.3690930, -0.5769094, 1.0379164, -1.2646537, 0.9460022

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3739734, upper bound: 3.3212949
time: 0.44 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3739734, upper bound: 3.3213076
time: 0.44 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.3088257, 0.4204044, -0.3722968, 0.5244855, -0.8333111, 0.7927012
1: -0.4969721, 0.6138726, -0.6036323, 0.7497613, -1.2467334, 1.2175047
2: -0.3497591, 0.6176917, -0.4169547, 0.7923486, -1.1421075, 1.0346463
3: -0.7490399, 0.8086340, -0.9431840, 0.9862554, -1.7352953, 1.7518177
4: -0.4839897, 0.8007323, -0.5769094, 1.0379164, -1.5219060, 1.3776417

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3739734, upper bound: 3.3212949
time: 0.46 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3739734, upper bound: 3.3213076
time: 0.50 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.1850611, 0.2096545, -0.6115561, 0.9495890, -1.1346501, 0.8212106
1: -0.2660490, 0.2913542, -1.0066755, 1.3170630, -1.5831120, 1.2980297
2: -0.2290718, 0.3008571, -0.6623885, 1.4479090, -1.6769807, 0.9632456
3: -0.3428191, 0.3601812, -1.7245955, 1.7292082, -2.0720272, 2.0847769
4: -0.2267372, 0.3690930, -1.0000154, 1.9054065, -2.1321437, 1.3691082

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3150157, upper bound: 3.3038922
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3149895, upper bound: 3.3067627
time: 0.45 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.3088257, 0.4204044, -0.6115561, 0.9495890, -1.2584147, 1.0319605
1: -0.4969721, 0.6138726, -1.0066755, 1.3170630, -1.8140351, 1.6205481
2: -0.3497591, 0.6176917, -0.6623885, 1.4479090, -1.7976677, 1.2800801
3: -0.7490399, 0.8086340, -1.7245955, 1.7292082, -2.4782476, 2.5332296
4: -0.4839897, 0.8007323, -1.0000154, 1.9054065, -2.3893962, 1.8007475

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3150157, upper bound: 3.3038922
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3149895, upper bound: 3.3067627
time: 0.46 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.2211066, 0.2775607, -0.1912085, 0.2109509, -0.4320576, 0.4687692
1: -0.3221700, 0.3887413, -0.2706243, 0.2982620, -0.6204320, 0.6593655
2: -0.2730529, 0.4024681, -0.2345207, 0.3197280, -0.5927809, 0.6369888
3: -0.4342754, 0.4909782, -0.3691734, 0.3672814, -0.8015568, 0.8601516
4: -0.2787128, 0.4985040, -0.2326370, 0.4024982, -0.6812110, 0.7311411

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1940192, upper bound: 3.3127877
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2076429, upper bound: 3.3366947
time: 0.45 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2076429, upper bound: 3.3366947
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.6227801, 0.8648329, -0.1912085, 0.2109509, -0.8337310, 1.0560414
1: -0.9949624, 1.2503651, -0.2706243, 0.2982620, -1.2932242, 1.5209894
2: -0.6539169, 1.3625121, -0.2345207, 0.3197280, -0.9736448, 1.5970329
3: -1.6873388, 1.6740556, -0.3691734, 0.3672814, -2.0546203, 2.0432291
4: -0.9924154, 1.8411534, -0.2326370, 0.4024982, -1.3949136, 2.0737906

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1940192, upper bound: 3.3127877
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2076429, upper bound: 3.3366947
time: 0.46 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2076429, upper bound: 3.3366947
time: 0.44 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.2211066, 0.2775607, -0.4519231, 0.6203508, -0.8414574, 0.7294837
1: -0.3221700, 0.3887413, -0.7154577, 0.8934209, -1.2155910, 1.1041989
2: -0.2730529, 0.4024681, -0.4850966, 0.9528847, -1.2259376, 0.8875644
3: -0.4342754, 0.4909782, -1.1693269, 1.1907833, -1.6250587, 1.6603050
4: -0.2787128, 0.4985040, -0.7152326, 1.2680461, -1.5467590, 1.2137365

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2018382, upper bound: 3.3297156
time: 0.44 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2076429, upper bound: 3.3739734
time: 0.45 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2076429, upper bound: 3.3739734
time: 0.45 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.6227801, 0.8648329, -0.4519231, 0.6203508, -1.2431309, 1.3167559
1: -0.9949624, 1.2503651, -0.7154577, 0.8934209, -1.8883834, 1.9658227
2: -0.6539169, 1.3625121, -0.4850966, 0.9528847, -1.6068015, 1.8476087
3: -1.6873388, 1.6740556, -1.1693269, 1.1907833, -2.8781221, 2.8433824
4: -0.9924154, 1.8411534, -0.7152326, 1.2680461, -2.2604616, 2.5563858

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2018382, upper bound: 3.3486306
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2076429, upper bound: 3.3590890
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2076429, upper bound: 3.3590890
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.7091317, 1.1715200, -0.1887316, 0.2037708, -0.9129025, 1.3602517
1: -1.1533722, 1.6230139, -0.2660364, 0.2885636, -1.4419354, 1.8890502
2: -0.7592056, 1.7569205, -0.2313113, 0.3093683, -1.0685740, 1.9882318
3: -2.0259933, 2.1003478, -0.3590131, 0.3537811, -2.3797743, 2.4593608
4: -1.1636406, 2.2863564, -0.2267287, 0.3890465, -1.5526870, 2.5130851

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1944766, upper bound: 3.2827253
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1944766, upper bound: 3.2827253
time: 0.45 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.4738564, 0.6639866, -0.1912085, 0.2109509, -0.6848074, 0.8551951
1: -0.7879710, 0.9146689, -0.2706243, 0.2982620, -1.0862329, 1.1852931
2: -0.5224863, 1.0312799, -0.2345207, 0.3197280, -0.8422142, 1.2658007
3: -1.3006039, 1.2264051, -0.3691734, 0.3672814, -1.6678851, 1.5955786
4: -0.7528779, 1.3891870, -0.2326370, 0.4024982, -1.1553757, 1.6218240

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1944632, upper bound: 3.2826990
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1944632, upper bound: 3.2826990
time: 0.45 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.7012182, 1.1544906, -0.4413537, 0.6057985, -1.3070164, 1.5958443
1: -1.1414101, 1.6001794, -0.6986523, 0.8728567, -2.0142667, 2.2988317
2: -0.7512843, 1.7325779, -0.4748410, 0.9287860, -1.6800702, 2.2074187
3: -2.0025437, 2.0722506, -1.1378615, 1.1622517, -3.1647954, 3.2101119
4: -1.1504493, 2.2558928, -0.6973975, 1.2340087, -2.3844576, 2.9532902

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2959703, upper bound: 3.3074373
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3038922, upper bound: 3.3150158
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3038922, upper bound: 3.3150157
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.4679187, 0.6531383, -0.4519231, 0.6203508, -1.0882694, 1.1050612
1: -0.7785782, 0.8996019, -0.7154577, 0.8934209, -1.6719987, 1.6150596
2: -0.5165930, 1.0138711, -0.4850966, 0.9528847, -1.4694777, 1.4989675
3: -1.2809948, 1.2070292, -1.1693269, 1.1907833, -2.4717779, 2.3763561
4: -0.7423084, 1.3666780, -0.7152326, 1.2680461, -2.0103545, 2.0819106

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2972614, upper bound: 3.3065539
time: 0.44 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3067627, upper bound: 3.3149895
time: 0.44 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3067627, upper bound: 3.3149895
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.2211066, 0.2775607, -0.3753012, 0.5288189, -0.7499256, 0.6528620
1: -0.3221700, 0.3887413, -0.6084703, 0.7558654, -1.0780354, 0.9972113
2: -0.2730529, 0.4024681, -0.4198778, 0.7994800, -1.0725329, 0.8223458
3: -0.4342754, 0.4909782, -0.9523419, 0.9947283, -1.4290037, 1.4433197
4: -0.2787128, 0.4985040, -0.5820951, 1.0479591, -1.3266720, 1.0805990

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3922350, upper bound: 3.3922350
time: 0.44 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3922350, upper bound: 3.3922350
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.6227801, 0.8648329, -0.3722968, 0.5244855, -1.1472656, 1.2371297
1: -0.9949624, 1.2503651, -0.6036323, 0.7497613, -1.7447236, 1.8539971
2: -0.6539169, 1.3625121, -0.4169547, 0.7923486, -1.4462652, 1.7794667
3: -1.6873388, 1.6740556, -0.9431840, 0.9862554, -2.6735942, 2.6172390
4: -0.9924154, 1.8411534, -0.5769094, 1.0379164, -2.0303319, 2.4180624

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3922350, upper bound: 3.3922350
time: 0.43 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3922350, upper bound: 3.3922350
time: 0.45 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.2211066, 0.2775607, -0.6188753, 0.9653776, -1.1864842, 0.8964360
1: -0.3221700, 0.3887413, -1.0178568, 1.3381594, -1.6603295, 1.4065980
2: -0.2730529, 0.4024681, -0.6695868, 1.4706283, -1.7436812, 1.0720547
3: -0.4342754, 0.4909782, -1.7465825, 1.7551888, -2.1894643, 2.2375603
4: -0.2787128, 0.4985040, -1.0122970, 1.9336305, -2.2123432, 1.5108011

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3261251, upper bound: 3.3695047
time: 0.45 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3262014, upper bound: 3.3686493
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.6227801, 0.8648329, -0.6115561, 0.9495890, -1.5723691, 1.4763889
1: -0.9949624, 1.2503651, -1.0066755, 1.3170630, -2.3120253, 2.2570405
2: -0.6539169, 1.3625121, -0.6623885, 1.4479090, -2.1018257, 2.0249004
3: -1.6873388, 1.6740556, -1.7245955, 1.7292082, -3.4165471, 3.3986506
4: -0.9924154, 1.8411534, -1.0000154, 1.9054065, -2.8978219, 2.8411682

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3260874, upper bound: 3.3696494
time: 0.47 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3261581, upper bound: 3.3684729
time: 0.44 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.4467238, 0.6215951, -0.3753012, 0.5288189, -0.9755427, 0.9968962
1: -0.7328915, 0.8427354, -0.6084703, 0.7558654, -1.4887569, 1.4512056
2: -0.4924225, 0.9656458, -0.4198778, 0.7994800, -1.2919024, 1.3855237
3: -1.2085794, 1.1289421, -0.9523419, 0.9947283, -2.2033076, 2.0812838
4: -0.7015048, 1.3040695, -0.5820951, 1.0479591, -1.7494637, 1.8861645

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3597859, upper bound: 3.3229110
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3597859, upper bound: 3.3229110
time: 0.45 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.4117214, 0.5673239, -0.3562311, 0.5022630, -0.9139844, 0.9235549
1: -0.6853573, 0.7768623, -0.5776588, 0.7177134, -1.4030706, 1.3545210
2: -0.4597958, 0.8666953, -0.4014583, 0.7561749, -1.2159708, 1.2681535
3: -1.0987918, 1.0422142, -0.8944356, 0.9425307, -2.0413225, 1.9366498
4: -0.6412443, 1.1624334, -0.5506753, 0.9870988, -1.6283431, 1.7131087

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3420087, upper bound: 3.3195359
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3420087, upper bound: 3.3195359
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.4467238, 0.6215951, -0.6188753, 0.9653776, -1.4121013, 1.2404703
1: -0.7328915, 0.8427354, -1.0178568, 1.3381594, -2.0710504, 1.8605920
2: -0.4924225, 0.9656458, -0.6695868, 1.4706283, -1.9630504, 1.6352326
3: -1.2085794, 1.1289421, -1.7465825, 1.7551888, -2.9637682, 2.8755245
4: -0.7015048, 1.3040695, -1.0122970, 1.9336305, -2.6351352, 2.3163664

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3132533, upper bound: 3.3132533
time: 0.50 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3132533, upper bound: 3.3133037
time: 0.51 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.4117214, 0.5673239, -0.5979920, 0.9179261, -1.3296475, 1.1653155
1: -0.6853573, 0.7768623, -0.9849302, 1.2731333, -1.9584905, 1.7617924
2: -0.4597958, 0.8666953, -0.6482026, 1.4037600, -1.8635558, 1.5148979
3: -1.0987918, 1.0422142, -1.6821562, 1.6755812, -2.7743731, 2.7243705
4: -0.6412443, 1.1624334, -0.9755487, 1.8512616, -2.4925060, 2.1379821

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3132533, upper bound: 3.3133380
time: 0.44 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3132533, upper bound: 3.3135253
time: 0.45 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.42 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 0, lower bound: -3.1566236, upper bound: 3.1566236
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 0, lower bound: -3.1566236, upper bound: 3.1566236
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 0, lower bound: -3.1566236, upper bound: 3.1566236
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 0, lower bound: -3.1566236, upper bound: 3.1566236
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 0, lower bound: -3.2674166, upper bound: 3.1889140
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 0, lower bound: -3.2674166, upper bound: 3.1889140
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 0, lower bound: -3.2674166, upper bound: 3.1889140
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 0, lower bound: -3.2674166, upper bound: 3.1889140
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 0, lower bound: -3.3056196, upper bound: 3.3043631
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 0, lower bound: -3.3087832, upper bound: 3.3054752
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 0, lower bound: -3.3067020, upper bound: 3.3087506
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 0, lower bound: -3.3098656, upper bound: 3.3098627
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 0, lower bound: -3.3043957, upper bound: 3.3043813
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 0, lower bound: -3.3043957, upper bound: 3.3054781
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 0, lower bound: -3.3053682, upper bound: 3.3086207
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 0, lower bound: -3.3053682, upper bound: 3.3098627
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 0, lower bound: -3.3366947, upper bound: 3.2076429
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 0, lower bound: -3.3366947, upper bound: 3.2076429
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 0, lower bound: -3.3366947, upper bound: 3.2076429
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 0, lower bound: -3.3366947, upper bound: 3.2076429
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 0, lower bound: -3.2827253, upper bound: 3.1944766
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 0, lower bound: -3.2826990, upper bound: 3.1944632
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 0, lower bound: -3.2827253, upper bound: 3.1944766
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 0, lower bound: -3.2826990, upper bound: 3.1944632
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 0, lower bound: -3.3739734, upper bound: 3.3212949
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 0, lower bound: -3.3739734, upper bound: 3.3213076
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 0, lower bound: -3.3739734, upper bound: 3.3212949
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 0, lower bound: -3.3739734, upper bound: 3.3213076
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 0, lower bound: -3.3150157, upper bound: 3.3038922
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 0, lower bound: -3.3149895, upper bound: 3.3067627
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 0, lower bound: -3.3150157, upper bound: 3.3038922
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 0, lower bound: -3.3149895, upper bound: 3.3067627
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 0, lower bound: -3.2076429, upper bound: 3.3366947
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 0, lower bound: -3.2076429, upper bound: 3.3366947
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 0, lower bound: -3.2076429, upper bound: 3.3366947
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 0, lower bound: -3.2076429, upper bound: 3.3366947
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 0, lower bound: -3.2076429, upper bound: 3.3739734
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 0, lower bound: -3.2076429, upper bound: 3.3739734
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 0, lower bound: -3.2076429, upper bound: 3.3590890
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 0, lower bound: -3.2076429, upper bound: 3.3590890
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 0, lower bound: -3.1944766, upper bound: 3.2827253
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 0, lower bound: -3.1944766, upper bound: 3.2827253
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 0, lower bound: -3.1944632, upper bound: 3.2826990
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 0, lower bound: -3.1944632, upper bound: 3.2826990
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 0, lower bound: -3.3038922, upper bound: 3.3150158
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 0, lower bound: -3.3038922, upper bound: 3.3150157
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 0, lower bound: -3.3067627, upper bound: 3.3149895
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 0, lower bound: -3.3067627, upper bound: 3.3149895
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 0, lower bound: -3.3922350, upper bound: 3.3922350
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 0, lower bound: -3.3922350, upper bound: 3.3922350
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 0, lower bound: -3.3922350, upper bound: 3.3922350
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 0, lower bound: -3.3922350, upper bound: 3.3922350
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 0, lower bound: -3.3261251, upper bound: 3.3695047
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 0, lower bound: -3.3262014, upper bound: 3.3686493
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 0, lower bound: -3.3260874, upper bound: 3.3696494
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 0, lower bound: -3.3261581, upper bound: 3.3684729
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 0, lower bound: -3.3597859, upper bound: 3.3229110
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 0, lower bound: -3.3597859, upper bound: 3.3229110
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 0, lower bound: -3.3420087, upper bound: 3.3195359
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 0, lower bound: -3.3420087, upper bound: 3.3195359
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 0, lower bound: -3.3132533, upper bound: 3.3132533
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 0, lower bound: -3.3132533, upper bound: 3.3133037
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 0, lower bound: -3.3132533, upper bound: 3.3133380
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 0, lower bound: -3.3132533, upper bound: 3.3135253

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0676768, 0.0203616, -0.0676768, 0.0203616, -0.0880384, 0.0880384
1: -0.0773866, 0.0391565, -0.0773866, 0.0391565, -0.1165432, 0.1165432
2: -0.0644345, 0.0282484, -0.0644345, 0.0282484, -0.0926829, 0.0926829
3: -0.0712000, 0.0471544, -0.0712000, 0.0471544, -0.1183544, 0.1183544
4: -0.0532835, 0.0354563, -0.0532835, 0.0354563, -0.0887398, 0.0887398

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1873232, upper bound: 3.2734669
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1898258, upper bound: 3.2772939
time: 0.39 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0676768, 0.0203616, -0.1649871, 0.1174907, -0.1851676, 0.1853487
1: -0.0773866, 0.0391565, -0.2215959, 0.1765458, -0.2539324, 0.2607524
2: -0.0644345, 0.0282484, -0.1996021, 0.1877475, -0.2521820, 0.2278505
3: -0.0712000, 0.0471544, -0.2914082, 0.2083298, -0.2795298, 0.3385626
4: -0.0532835, 0.0354563, -0.1816220, 0.2294925, -0.2827760, 0.2170783

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1873232, upper bound: 3.2734669
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1898258, upper bound: 3.2772939
time: 0.38 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.1649871, 0.1174907, -0.0676768, 0.0203616, -0.1853487, 0.1851676
1: -0.2215959, 0.1765458, -0.0773866, 0.0391565, -0.2607524, 0.2539324
2: -0.1996021, 0.1877475, -0.0644345, 0.0282484, -0.2278505, 0.2521820
3: -0.2914082, 0.2083298, -0.0712000, 0.0471544, -0.3385626, 0.2795298
4: -0.1816220, 0.2294925, -0.0532835, 0.0354563, -0.2170783, 0.2827760

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1076827, upper bound: 3.0515106
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1566236, upper bound: 3.1566236
time: 0.39 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.1649871, 0.1174907, -0.1649871, 0.1174907, -0.2824779, 0.2824779
1: -0.2215959, 0.1765458, -0.2215959, 0.1765458, -0.3981417, 0.3981417
2: -0.1996021, 0.1877475, -0.1996021, 0.1877475, -0.3873496, 0.3873496
3: -0.2914082, 0.2083298, -0.2914082, 0.2083298, -0.4997380, 0.4997380
4: -0.1816220, 0.2294925, -0.1816220, 0.2294925, -0.4111145, 0.4111145

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1076827, upper bound: 3.0515106
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1566236, upper bound: 3.1566236
time: 0.37 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0676768, 0.0203616, -0.1850611, 0.2096545, -0.2773313, 0.2054227
1: -0.0773866, 0.0391565, -0.2660490, 0.2913542, -0.3687408, 0.3052056
2: -0.0644345, 0.0282484, -0.2290718, 0.3008571, -0.3652916, 0.2573202
3: -0.0712000, 0.0471544, -0.3428191, 0.3601812, -0.4313812, 0.3899735
4: -0.0532835, 0.0354563, -0.2267372, 0.3690930, -0.4223765, 0.2621935

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2589558, upper bound: 3.2980102
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3057312, upper bound: 3.3148614
time: 0.40 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0676768, 0.0203616, -0.3088257, 0.4204044, -0.4880812, 0.3291872
1: -0.0773866, 0.0391565, -0.4969721, 0.6138726, -0.6912593, 0.5361286
2: -0.0644345, 0.0282484, -0.3497591, 0.6176917, -0.6821262, 0.3780075
3: -0.0712000, 0.0471544, -0.7490399, 0.8086340, -0.8798340, 0.7961944
4: -0.0532835, 0.0354563, -0.4839897, 0.8007323, -0.8540158, 0.5194458

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2589558, upper bound: 3.2980102
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3057312, upper bound: 3.3148614
time: 0.42 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.1649871, 0.1174907, -0.1850611, 0.2096545, -0.3746416, 0.3025518
1: -0.2215959, 0.1765458, -0.2660490, 0.2913542, -0.5129501, 0.4425948
2: -0.1996021, 0.1877475, -0.2290718, 0.3008571, -0.5004592, 0.4168193
3: -0.2914082, 0.2083298, -0.3428191, 0.3601812, -0.6515895, 0.5511489
4: -0.1816220, 0.2294925, -0.2267372, 0.3690930, -0.5507150, 0.4562297

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1942296, upper bound: 3.0787074
time: 0.43 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2674166, upper bound: 3.1889140
time: 0.43 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.1649871, 0.1174907, -0.3088257, 0.4204044, -0.5853915, 0.4263164
1: -0.2215959, 0.1765458, -0.4969721, 0.6138726, -0.8354685, 0.6735178
2: -0.1996021, 0.1877475, -0.3497591, 0.6176917, -0.8172938, 0.5375066
3: -0.2914082, 0.2083298, -0.7490399, 0.8086340, -1.1000422, 0.9573698
4: -0.1816220, 0.2294925, -0.4839897, 0.8007323, -0.9823543, 0.7134821

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1942296, upper bound: 3.0787074
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2674166, upper bound: 3.1889140
time: 0.44 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.2419151, 0.3237536, -0.1569118, 0.1010650, -0.3429801, 0.4806654
1: -0.3723261, 0.4576313, -0.2085741, 0.1534233, -0.5257494, 0.6662055
2: -0.2878267, 0.4800687, -0.1883991, 0.1622353, -0.4500620, 0.6684678
3: -0.5494078, 0.5889983, -0.2698923, 0.1826724, -0.7320802, 0.8588907
4: -0.3522730, 0.6066738, -0.1733955, 0.1943219, -0.5465949, 0.7800693

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2848344, upper bound: 3.2915910
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2993443, upper bound: 3.3017473
time: 0.43 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 17

Time for candidate selection: 5.02 seconds

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2908436, upper bound: 3.2810768
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2958534, upper bound: 3.2990657
time: 0.42 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.2559387, 0.3415756, -0.1805636, 0.1780242, -0.4339629, 0.5221391
1: -0.3946697, 0.4863652, -0.2509359, 0.2545417, -0.6492113, 0.7373011
2: -0.2981569, 0.5062003, -0.2206551, 0.2722895, -0.5704464, 0.7268554
3: -0.5855743, 0.6284050, -0.3329057, 0.3066460, -0.8922204, 0.9613106
4: -0.3760676, 0.6415557, -0.2074082, 0.3399702, -0.7160378, 0.8489639

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2807560, upper bound: 3.2910326
time: 0.45 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2821077, upper bound: 3.2981584
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 17

Time for candidate selection: 5.10 seconds

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2959452, upper bound: 3.2826198
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3009550, upper bound: 3.3006087
time: 0.42 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.3905965, 0.5407779, -0.1549339, 0.0947623, -0.4853587, 0.6957119
1: -0.6246507, 0.7617285, -0.2046678, 0.1448168, -0.7694674, 0.9663963
2: -0.4283231, 0.8243151, -0.1857630, 0.1522939, -0.5806168, 1.0100781
3: -1.0045357, 1.0123587, -0.2646994, 0.1729024, -1.1774379, 1.2770581
4: -0.6061574, 1.0865963, -0.1709704, 0.1812100, -0.7873673, 1.2575666

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3004120, upper bound: 3.3060811
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2857186, upper bound: 3.2953114
time: 0.43 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 17

Time for candidate selection: 5.06 seconds

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3067020, upper bound: 3.3087506
time: 0.44 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2914924, upper bound: 3.2851578
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2960610, upper bound: 3.3000349
time: 0.46 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.4087062, 0.5647650, -0.1763318, 0.1661702, -0.5748761, 0.7410968
1: -0.6539204, 0.7966581, -0.2437494, 0.2383934, -0.8923137, 1.0404074
2: -0.4457314, 0.8629469, -0.2151669, 0.2555566, -0.7012880, 1.0781138
3: -1.0570911, 1.0608249, -0.3222067, 0.2850041, -1.3420951, 1.3830316
4: -0.6360155, 1.1408229, -0.1993417, 0.3180806, -0.9540962, 1.3401647

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2831742, upper bound: 3.3024922
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2816402, upper bound: 3.2947530
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 17

Time for candidate selection: 5.14 seconds

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3098656, upper bound: 3.3098627
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2965940, upper bound: 3.2867008
time: 0.44 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3011626, upper bound: 3.3015779
time: 0.42 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.2559387, 0.3415756, -0.2559387, 0.3415756, -0.5975143, 0.5975143
1: -0.3946697, 0.4863652, -0.3946697, 0.4863652, -0.8810349, 0.8810349
2: -0.2981569, 0.5062003, -0.2981569, 0.5062003, -0.8043572, 0.8043572
3: -0.5855743, 0.6284050, -0.5855743, 0.6284050, -1.2139792, 1.2139792
4: -0.3760676, 0.6415557, -0.3760676, 0.6415557, -1.0176232, 1.0176232

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 17

Time for candidate selection: 4.25 seconds

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2928931, upper bound: 3.2813966
time: 0.44 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3006618, upper bound: 3.3006618
time: 0.42 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.2559387, 0.3415756, -0.4087062, 0.5647650, -0.8207036, 0.7502818
1: -0.3946697, 0.4863652, -0.6539204, 0.7966581, -1.1913278, 1.1402855
2: -0.2981569, 0.5062003, -0.4457314, 0.8629469, -1.1611037, 0.9519317
3: -0.5855743, 0.6284050, -1.0570911, 1.0608249, -1.6463993, 1.6854960
4: -0.3760676, 0.6415557, -0.6360155, 1.1408229, -1.5168905, 1.2775711

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 17

Time for candidate selection: 4.37 seconds

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2928931, upper bound: 3.2826198
time: 0.44 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3006618, upper bound: 3.3008694
time: 0.43 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.4087062, 0.5647650, -0.2559387, 0.3415756, -0.7502818, 0.8207036
1: -0.6539204, 0.7966581, -0.3946697, 0.4863652, -1.1402856, 1.1913278
2: -0.4457314, 0.8629469, -0.2981569, 0.5062003, -0.9519317, 1.1611037
3: -1.0570911, 1.0608249, -0.5855743, 0.6284050, -1.6854960, 1.6463993
4: -0.6360155, 1.1408229, -0.3760676, 0.6415557, -1.2775711, 1.5168905

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 17

Time for candidate selection: 4.39 seconds

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2925831, upper bound: 3.2854776
time: 0.43 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3006618, upper bound: 3.3014515
time: 0.42 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.4087062, 0.5647650, -0.4087062, 0.5647650, -0.9734710, 0.9734709
1: -0.6539204, 0.7966581, -0.6539204, 0.7966581, -1.4505785, 1.4505785
2: -0.4457314, 0.8629469, -0.4457314, 0.8629469, -1.3086783, 1.3086783
3: -1.0570911, 1.0608249, -1.0570911, 1.0608249, -2.1179161, 2.1179159
4: -0.6360155, 1.1408229, -0.6360155, 1.1408229, -1.7768379, 1.7768378

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 17

Time for candidate selection: 4.36 seconds

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2925831, upper bound: 3.2866757
time: 0.46 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3006618, upper bound: 3.3018006
time: 0.49 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0676768, 0.0203616, -0.2211066, 0.2775607, -0.3452376, 0.2414682
1: -0.0773866, 0.0391565, -0.3221700, 0.3887413, -0.4661279, 0.3613265
2: -0.0644345, 0.0282484, -0.2730529, 0.4024681, -0.4669026, 0.3013013
3: -0.0712000, 0.0471544, -0.4342754, 0.4909782, -0.5621781, 0.4814299
4: -0.0532835, 0.0354563, -0.2787128, 0.4985040, -0.5517876, 0.3141691

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3502519, upper bound: 3.3192499
time: 0.43 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3459156, upper bound: 3.3044647
time: 0.42 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0676768, 0.0203616, -0.6227801, 0.8648329, -0.9325097, 0.6431416
1: -0.0773866, 0.0391565, -0.9949624, 1.2503651, -1.3277518, 1.0341187
2: -0.0644345, 0.0282484, -0.6539169, 1.3625121, -1.4269466, 0.6821653
3: -0.0712000, 0.0471544, -1.6873388, 1.6740556, -1.7452556, 1.7344933
4: -0.0532835, 0.0354563, -0.9924154, 1.8411534, -1.8944370, 1.0278717

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3502519, upper bound: 3.3192499
time: 0.45 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3459156, upper bound: 3.3044647
time: 0.44 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.1649871, 0.1174907, -0.2211066, 0.2775607, -0.4425478, 0.3385974
1: -0.2215959, 0.1765458, -0.3221700, 0.3887413, -0.6103371, 0.4987158
2: -0.1996021, 0.1877475, -0.2730529, 0.4024681, -0.6020702, 0.4608004
3: -0.2914082, 0.2083298, -0.4342754, 0.4909782, -0.7823864, 0.6426053
4: -0.1816220, 0.2294925, -0.2787128, 0.4985040, -0.6801261, 0.5082053

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3127877, upper bound: 3.1940192
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3012218, upper bound: 3.1067568
time: 0.44 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3366947, upper bound: 3.2076383
time: 0.46 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.1649871, 0.1174907, -0.6227801, 0.8648329, -1.0298200, 0.7402708
1: -0.2215959, 0.1765458, -0.9949624, 1.2503651, -1.4719610, 1.1715081
2: -0.1996021, 0.1877475, -0.6539169, 1.3625121, -1.5621142, 0.8416644
3: -0.2914082, 0.2083298, -1.6873388, 1.6740556, -1.9654638, 1.8956686
4: -0.1816220, 0.2294925, -0.9924154, 1.8411534, -2.0227754, 1.2219079

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3127877, upper bound: 3.1940192
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3012218, upper bound: 3.1067568
time: 0.45 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3366947, upper bound: 3.2076383
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0674032, 0.0186966, -0.7091317, 1.1715200, -1.2389232, 0.7278283
1: -0.0768805, 0.0363743, -1.1533722, 1.6230139, -1.6998943, 1.1897465
2: -0.0640822, 0.0255594, -0.7592056, 1.7569205, -1.8210027, 0.7847650
3: -0.0705857, 0.0442395, -2.0259933, 2.1003478, -2.1709335, 2.0702329
4: -0.0525095, 0.0321029, -1.1636406, 2.2863564, -2.3388660, 1.1957436

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3154066, upper bound: 3.3091536
time: 0.43 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3101665, upper bound: 3.2864966
time: 0.42 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0676768, 0.0203616, -0.4738564, 0.6639866, -0.7316635, 0.4942180
1: -0.0773866, 0.0391565, -0.7879710, 0.9146689, -0.9920555, 0.8271275
2: -0.0644345, 0.0282484, -0.5224863, 1.0312799, -1.0957144, 0.5507348
3: -0.0712000, 0.0471544, -1.3006039, 1.2264051, -1.2976052, 1.3477583
4: -0.0532835, 0.0354563, -0.7528779, 1.3891870, -1.4424706, 0.7883341

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3153851, upper bound: 3.3165742
time: 0.44 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3101450, upper bound: 3.2864966
time: 0.47 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.1636999, 0.1126830, -0.7091317, 1.1715200, -1.3352199, 0.8218147
1: -0.2191397, 0.1704445, -1.1533722, 1.6230139, -1.8421535, 1.3238165
2: -0.1978667, 0.1806396, -0.7592056, 1.7569205, -1.9547871, 0.9398452
3: -0.2878307, 0.2010850, -2.0259933, 2.1003478, -2.3881783, 2.2270784
4: -0.1800120, 0.2199385, -1.1636406, 2.2863564, -2.4663684, 1.3835791

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2753965, upper bound: 3.1892053
time: 0.39 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2813175, upper bound: 3.1911572
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2821828, upper bound: 3.1934300
time: 0.42 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.1649871, 0.1174907, -0.4738564, 0.6639866, -0.8289738, 0.5913472
1: -0.2215959, 0.1765458, -0.7879710, 0.9146689, -1.1362647, 0.9645168
2: -0.1996021, 0.1877475, -0.5224863, 1.0312799, -1.2308820, 0.7102338
3: -0.2914082, 0.2083298, -1.3006039, 1.2264051, -1.5178133, 1.5089337
4: -0.1816220, 0.2294925, -0.7528779, 1.3891870, -1.5708090, 0.9823701

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2744161, upper bound: 3.1889506
time: 0.43 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2812960, upper bound: 3.1911572
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2821613, upper bound: 3.1934300
time: 0.43 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1850611, 0.2096545, -0.2211066, 0.2775607, -0.4626218, 0.4307611
1: -0.2660490, 0.2913542, -0.3221700, 0.3887413, -0.6547903, 0.6135242
2: -0.2290718, 0.3008571, -0.2730529, 0.4024681, -0.6315399, 0.5739100
3: -0.3428191, 0.3601812, -0.4342754, 0.4909782, -0.8337973, 0.7944567
4: -0.2267372, 0.3690930, -0.2787128, 0.4985040, -0.7252413, 0.6478058

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3503717, upper bound: 3.3192499
time: 0.43 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3685972, upper bound: 3.3207771
time: 0.43 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3694722, upper bound: 3.3251339
time: 0.43 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1850611, 0.2096545, -0.6227801, 0.8648329, -1.0498940, 0.8324345
1: -0.2660490, 0.2913542, -0.9949624, 1.2503651, -1.5164142, 1.2863165
2: -0.2290718, 0.3008571, -0.6539169, 1.3625121, -1.5915840, 0.9547737
3: -0.3428191, 0.3601812, -1.6873388, 1.6740556, -2.0168748, 2.0475202
4: -0.2267372, 0.3690930, -0.9924154, 1.8411534, -2.0678906, 1.3615084

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3503717, upper bound: 3.3192499
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3685972, upper bound: 3.3207771
time: 0.43 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3694723, upper bound: 3.3251339
time: 0.45 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.3088257, 0.4204044, -0.2211066, 0.2775607, -0.5863864, 0.6415110
1: -0.4969721, 0.6138726, -0.3221700, 0.3887413, -0.8857133, 0.9360427
2: -0.3497591, 0.6176917, -0.2730529, 0.4024681, -0.7522272, 0.8907446
3: -0.7490399, 0.8086340, -0.4342754, 0.4909782, -1.2400180, 1.2429094
4: -0.4839897, 0.8007323, -0.2787128, 0.4985040, -0.9824935, 1.0794451

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3666067, upper bound: 3.3127592
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3585045, upper bound: 3.2762150
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3725498, upper bound: 3.3208464
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.3088257, 0.4204044, -0.6227801, 0.8648329, -1.1736586, 1.0431845
1: -0.4969721, 0.6138726, -0.9949624, 1.2503651, -1.7473372, 1.6088350
2: -0.3497591, 0.6176917, -0.6539169, 1.3625121, -1.7122711, 1.2716085
3: -0.7490399, 0.8086340, -1.6873388, 1.6740556, -2.4230950, 2.4959729
4: -0.4839897, 0.8007323, -0.9924154, 1.8411534, -2.3251426, 1.7931477

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3666068, upper bound: 3.3127592
time: 0.46 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3585045, upper bound: 3.2762150
time: 0.44 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3725498, upper bound: 3.3208464
time: 0.44 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.1827936, 0.2018988, -0.7012182, 1.1544906, -1.3372842, 0.9031171
1: -0.2616701, 0.2801387, -1.1414101, 1.6001794, -1.8618495, 1.4215486
2: -0.2258358, 0.2901761, -0.7512843, 1.7325779, -1.9584137, 1.0414604
3: -0.3367829, 0.3451592, -2.0025437, 2.0722506, -2.4090335, 2.3477025
4: -0.2213144, 0.3556997, -1.1504493, 2.2558928, -2.4772072, 1.5061489

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3087074, upper bound: 3.3015780
time: 0.44 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3087158, upper bound: 3.3027385
time: 0.43 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.1850611, 0.2096545, -0.4679187, 0.6531383, -0.8381994, 0.6775731
1: -0.2660490, 0.2913542, -0.7785782, 0.8996019, -1.1656508, 1.0699321
2: -0.2290718, 0.3008571, -0.5165930, 1.0138711, -1.2429428, 0.8174500
3: -0.3428191, 0.3601812, -1.2809948, 1.2070292, -1.5498483, 1.6411760
4: -0.2267372, 0.3690930, -0.7423084, 1.3666780, -1.5934153, 1.1114012

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28

Time for candidate selection: 0.19 seconds
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=3.5238394737243652
rel_dist={0: [-3.3982802470538505, 3.398280247053849]}

## Binary search (step 1) starts
Candidate diff: 0.1250000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3759131, upper bound: 3.3322402
time: 0.42 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3944676, upper bound: 3.3944676
time: 0.42 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.02 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.02
Output dim: 0, lower bound: -3.3759131, upper bound: 3.3322402
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.02
Output dim: 0, lower bound: -3.3944676, upper bound: 3.3944676

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.2895984, 0.3943615, -1.2466472, 2.2771921, -2.5667903, 1.6410087
1: -0.4578271, 0.5641531, -1.9637374, 3.1950235, -3.6528506, 2.5278900
2: -0.3370600, 0.5814233, -1.3604455, 3.2710245, -3.6080840, 1.9418688
3: -0.6758766, 0.7399411, -3.4595599, 4.0564485, -4.7323251, 4.1995006
4: -0.4528955, 0.7513722, -2.1785955, 4.2066536, -4.6595483, 2.9299674

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3169978, upper bound: 3.3169978
time: 0.46 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3169978, upper bound: 3.3322402
time: 0.43 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -1.0333478, 1.8749698, -1.1023791, 1.9804223, -3.0137699, 2.9773486
1: -1.6493901, 2.6237774, -1.7432088, 2.7915106, -4.4409008, 4.3669853
2: -1.1160958, 2.7230368, -1.1966579, 2.8632312, -3.9793260, 3.9196947
3: -2.9354391, 3.3341267, -3.0622921, 3.5597639, -6.4952030, 6.3964176
4: -1.7549448, 3.5216486, -1.9117329, 3.6996984, -5.4546423, 5.4333816

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3322402, upper bound: 3.3759131
time: 0.41 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3322402, upper bound: 3.3944677
time: 0.52 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.35 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.35
Output dim: 0, lower bound: -3.3169978, upper bound: 3.3169978
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.35
Output dim: 0, lower bound: -3.3169978, upper bound: 3.3322402
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.35
Output dim: 0, lower bound: -3.3322402, upper bound: 3.3759131
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.35
Output dim: 0, lower bound: -3.3322402, upper bound: 3.3944677

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.2895984, 0.3943615, -0.2895984, 0.3943615, -0.6839598, 0.6839598
1: -0.4578271, 0.5641531, -0.4578271, 0.5641531, -1.0219798, 1.0219798
2: -0.3370600, 0.5814233, -0.3370600, 0.5814233, -0.9184830, 0.9184831
3: -0.6758766, 0.7399411, -0.6758766, 0.7399411, -1.4158176, 1.4158175
4: -0.4528955, 0.7513722, -0.4528955, 0.7513722, -1.2042676, 1.2042676

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3169505, upper bound: 3.3169978
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3169978, upper bound: 3.3169978
time: 0.39 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.2895984, 0.3943615, -1.0333478, 1.8749698, -2.1645675, 1.4277093
1: -0.4578271, 0.5641531, -1.6493901, 2.6237774, -3.0816045, 2.2135432
2: -0.3370600, 0.5814233, -1.1160958, 2.7230368, -3.0600965, 1.6975191
3: -0.6758766, 0.7399411, -2.9354391, 3.3341267, -4.0100031, 3.6753800
4: -0.4528955, 0.7513722, -1.7549448, 3.5216486, -3.9745436, 2.5063167

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3169505, upper bound: 3.3322402
time: 0.45 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3169978, upper bound: 3.3322402
time: 0.44 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -1.0333478, 1.8749698, -0.2895984, 0.3943615, -1.4277093, 2.1645677
1: -1.6493901, 2.6237774, -0.4578271, 0.5641531, -2.2135432, 3.0816045
2: -1.1160958, 2.7230368, -0.3370600, 0.5814233, -1.6975191, 3.0600965
3: -2.9354391, 3.3341267, -0.6758766, 0.7399411, -3.6753798, 4.0100031
4: -1.7549448, 3.5216486, -0.4528955, 0.7513722, -2.5063167, 3.9745436

Time for backsubstitution: 2.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3189540, upper bound: 3.3648028
time: 0.51 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3052940, upper bound: 3.3127344
time: 0.44 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -1.0333478, 1.8749698, -1.0333478, 1.8749698, -2.9083173, 2.9083176
1: -1.6493901, 2.6237774, -1.6493901, 2.6237774, -4.2731667, 4.2731667
2: -1.1160958, 2.7230368, -1.1160958, 2.7230368, -3.8391325, 3.8391325
3: -2.9354391, 3.3341267, -2.9354391, 3.3341267, -6.2695656, 6.2695656
4: -1.7549448, 3.5216486, -1.7549448, 3.5216486, -5.2765932, 5.2765932

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3189540, upper bound: 3.3674661
time: 0.43 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3052940, upper bound: 3.3205784
time: 0.44 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.36 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.36
Output dim: 0, lower bound: -3.3169505, upper bound: 3.3169978
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.36
Output dim: 0, lower bound: -3.3169978, upper bound: 3.3169978
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.36
Output dim: 0, lower bound: -3.3169505, upper bound: 3.3322402
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.36
Output dim: 0, lower bound: -3.3169978, upper bound: 3.3322402
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.36
Output dim: 0, lower bound: -3.3189540, upper bound: 3.3648028
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.36
Output dim: 0, lower bound: -3.3052940, upper bound: 3.3127344
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.36
Output dim: 0, lower bound: -3.3189540, upper bound: 3.3674661
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.36
Output dim: 0, lower bound: -3.3052940, upper bound: 3.3205784

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1912085, 0.2109509, -0.2417870, 0.3204530, -0.5116615, 0.4527379
1: -0.2706243, 0.2982620, -0.3761925, 0.4566836, -0.7273078, 0.6744545
2: -0.2345207, 0.3197280, -0.2893954, 0.4721112, -0.7066320, 0.6091233
3: -0.3691734, 0.3672814, -0.5346558, 0.5924658, -0.9616393, 0.9019372
4: -0.2326370, 0.4024982, -0.3649140, 0.6016475, -0.8342845, 0.7674121

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3169505, upper bound: 3.3169505
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3169505, upper bound: 3.3169978
time: 0.40 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.4519231, 0.6203508, -0.2421183, 0.3199053, -0.7718282, 0.8624691
1: -0.7154577, 0.8934209, -0.3783879, 0.4577627, -1.1732204, 1.2718089
2: -0.4850966, 0.9528847, -0.2893090, 0.4687606, -0.9538568, 1.2421937
3: -1.1693269, 1.1907833, -0.5346802, 0.5924149, -1.7617416, 1.7254634
4: -0.7152326, 1.2680461, -0.3628546, 0.5959299, -1.3111625, 1.6309007

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3169978, upper bound: 3.3169505
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3169978, upper bound: 3.3169978
time: 0.44 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.1912085, 0.2109509, -0.9301875, 1.6475395, -1.8387480, 1.1411384
1: -0.2706243, 0.2982620, -1.4871773, 2.3137052, -2.5843294, 1.7854393
2: -0.2345207, 0.3197280, -0.9956113, 2.4103410, -2.6448617, 1.3153392
3: -0.3691734, 0.3672814, -2.6399546, 2.9517164, -3.3208899, 3.0072360
4: -0.2326370, 0.4024982, -1.5579123, 3.1323736, -3.3650107, 1.9604105

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3147914, upper bound: 3.2024730
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2347830, upper bound: 3.1805324
time: 0.40 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.4519231, 0.6203508, -0.8773195, 1.5370352, -1.9889574, 1.4976702
1: -0.7154577, 0.8934209, -1.4102795, 2.1756563, -2.8911140, 2.3037004
2: -0.4850966, 0.9528847, -0.9419119, 2.2478163, -2.7329125, 1.8947966
3: -1.1693269, 1.1907833, -2.4848533, 2.7805126, -3.9498396, 3.6756365
4: -0.7152326, 1.2680461, -1.4737182, 2.9270277, -3.6422603, 2.7417643

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3648028, upper bound: 3.3189540
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3127344, upper bound: 3.3052940
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.3753012, 0.5288189, -0.2500562, 0.3355049, -0.7108061, 0.7788751
1: -0.6084703, 0.7558654, -0.3929735, 0.4804459, -1.0889161, 1.1488390
2: -0.4198778, 0.7994800, -0.2982292, 0.4920845, -0.9119623, 1.0977092
3: -0.9523419, 0.9947283, -0.5602974, 0.6253927, -1.5777346, 1.5550256
4: -0.5820951, 1.0479591, -0.3831701, 0.6284484, -1.2105432, 1.4311292

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2024730, upper bound: 3.3147914
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3189540, upper bound: 3.3648028
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.6188753, 0.9653776, -0.2297899, 0.3003063, -0.9191813, 1.1951674
1: -1.0178568, 1.3381594, -0.3553332, 0.4281315, -1.4459882, 1.6934927
2: -0.6695868, 1.4706283, -0.2772543, 0.4439168, -1.1135037, 1.7478825
3: -1.7465825, 1.7551888, -0.5033398, 0.5527423, -2.2993245, 2.2585287
4: -1.0122970, 1.9336305, -0.3434171, 0.5634758, -1.5757726, 2.2770476

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1805324, upper bound: 3.2347830
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3052940, upper bound: 3.3127344
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.3753012, 0.5288189, -0.9479318, 1.6877630, -2.0630641, 1.4767506
1: -0.6084703, 0.7558654, -1.5172229, 2.3689971, -2.9774673, 2.2730882
2: -0.4198778, 0.7994800, -1.0184590, 2.4641125, -2.8839903, 1.8179386
3: -0.9523419, 0.9947283, -2.6902966, 3.0217123, -3.9740541, 3.6850233
4: -0.5820951, 1.0479591, -1.5967597, 3.2003412, -3.7824364, 2.6447184

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3205783, upper bound: 3.3205783
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3205783, upper bound: 3.3205783
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.6188753, 0.9653776, -0.8875108, 1.5558972, -2.1747720, 1.8528882
1: -1.0178568, 1.3381594, -1.4290922, 2.1793237, -3.1971803, 2.7672515
2: -0.6695868, 1.4706283, -0.9549034, 2.2824402, -2.9520268, 2.4255316
3: -1.7465825, 1.7551888, -2.5197053, 2.7901378, -4.5367203, 4.2748938
4: -1.0122970, 1.9336305, -1.4910679, 2.9670417, -3.9793386, 3.4246984

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3205783, upper bound: 3.3205783
time: 0.43 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3205783, upper bound: 3.3205783
time: 0.40 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.11 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 0, lower bound: -3.3169505, upper bound: 3.3169505
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 0, lower bound: -3.3169505, upper bound: 3.3169978
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 0, lower bound: -3.3169978, upper bound: 3.3169505
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 0, lower bound: -3.3169978, upper bound: 3.3169978
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 0, lower bound: -3.3147914, upper bound: 3.2024730
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 0, lower bound: -3.2347830, upper bound: 3.1805324
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 0, lower bound: -3.3648028, upper bound: 3.3189540
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 0, lower bound: -3.3127344, upper bound: 3.3052940
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 0, lower bound: -3.2024730, upper bound: 3.3147914
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 0, lower bound: -3.3189540, upper bound: 3.3648028
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 0, lower bound: -3.1805324, upper bound: 3.2347830
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 0, lower bound: -3.3052940, upper bound: 3.3127344
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 0, lower bound: -3.3205783, upper bound: 3.3205783
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 0, lower bound: -3.3205783, upper bound: 3.3205783
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 0, lower bound: -3.3205783, upper bound: 3.3205783
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.11
Output dim: 0, lower bound: -3.3205783, upper bound: 3.3205783

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1912085, 0.2109509, -0.1912085, 0.2109509, -0.4021594, 0.4021594
1: -0.2706243, 0.2982620, -0.2706243, 0.2982620, -0.5688863, 0.5688863
2: -0.2345207, 0.3197280, -0.2345207, 0.3197280, -0.5542487, 0.5542487
3: -0.3691734, 0.3672814, -0.3691734, 0.3672814, -0.7364548, 0.7364548
4: -0.2326370, 0.4024982, -0.2326370, 0.4024982, -0.6351352, 0.6351352

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1870526, upper bound: 3.2741617
time: 0.43 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1566236, upper bound: 3.1566236
time: 0.40 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1912085, 0.2109509, -0.4002510, 0.5594531, -0.7506616, 0.6112019
1: -0.2706243, 0.2982620, -0.6648765, 0.8165386, -1.0871629, 0.9631385
2: -0.2345207, 0.3197280, -0.4584243, 0.7870062, -1.0215269, 0.7781522
3: -0.3691734, 0.3672814, -0.9596827, 1.0986850, -1.4678584, 1.3269641
4: -0.2326370, 0.4024982, -0.6609793, 1.0315057, -1.2641428, 1.0634774

Time for backsubstitution: 2.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1870526, upper bound: 3.2987063
time: 0.44 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1566236, upper bound: 3.1816649
time: 0.44 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.4519231, 0.6203508, -0.1912085, 0.2109509, -0.6628739, 0.8115593
1: -0.7154577, 0.8934209, -0.2706243, 0.2982620, -1.0137197, 1.1640452
2: -0.4850966, 0.9528847, -0.2345207, 0.3197280, -0.8048245, 1.1874055
3: -1.1693269, 1.1907833, -0.3691734, 0.3672814, -1.5366082, 1.5599567
4: -0.7152326, 1.2680461, -0.2326370, 0.4024982, -1.1177306, 1.5006832

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3087832, upper bound: 3.3054649
time: 0.46 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3098656, upper bound: 3.3098627
time: 0.42 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.4519231, 0.6203508, -0.4002510, 0.5594531, -1.0113761, 1.0206017
1: -0.7154577, 0.8934209, -0.6648765, 0.8165386, -1.5319963, 1.5582974
2: -0.4850966, 0.9528847, -0.4584243, 0.7870062, -1.2721024, 1.4113090
3: -1.1693269, 1.1907833, -0.9596827, 1.0986850, -2.2680120, 2.1504660
4: -0.7152326, 1.2680461, -0.6609793, 1.0315057, -1.7467382, 1.9290254

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3087832, upper bound: 3.3054678
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3098656, upper bound: 3.3098627
time: 0.40 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.1803700, 0.1815150, -0.3100591, 0.4353619, -0.6157318, 0.4915742
1: -0.2514807, 0.2583605, -0.5008603, 0.6214672, -0.8729479, 0.7592206
2: -0.2205296, 0.2771387, -0.3562234, 0.6509233, -0.8714529, 0.6333621
3: -0.3344405, 0.3125712, -0.7530009, 0.8102221, -1.1446626, 1.0655720
4: -0.2086931, 0.3468166, -0.4713841, 0.8378688, -1.0465620, 0.8182006

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2980723, upper bound: 3.1901709
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3099443, upper bound: 3.1973792
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2347830, upper bound: 3.1805324
time: 0.43 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2347830, upper bound: 3.1805324
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.1752371, 0.1600563, -0.5475030, 0.8051696, -0.9804068, 0.7075592
1: -0.2410980, 0.2296271, -0.9047095, 1.1165475, -1.3576455, 1.1343366
2: -0.2133930, 0.2479965, -0.5974170, 1.2467538, -1.4601468, 0.8454133
3: -0.3201365, 0.2728859, -1.5290978, 1.4838481, -1.8039846, 1.8019837
4: -0.1963671, 0.3093478, -0.8865705, 1.6566507, -1.8530178, 1.1959183

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2347830, upper bound: 3.1805324
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2347830, upper bound: 3.1805324
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.4152186, 0.5710753, -0.3320688, 0.4660973, -0.8813158, 0.9031439
1: -0.6577345, 0.8221928, -0.5381686, 0.6677936, -1.3255280, 1.3603611
2: -0.4497601, 0.8711857, -0.3777248, 0.6980434, -1.1478035, 1.2489104
3: -1.0622311, 1.0926554, -0.8200477, 0.8725698, -1.9348007, 1.9127032
4: -0.6544805, 1.1535711, -0.5075729, 0.9035305, -1.5580109, 1.6611438

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3558204, upper bound: 3.3100745
time: 0.46 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3127344, upper bound: 3.3052940
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3127344, upper bound: 3.3052940
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.3734080, 0.5128634, -0.5010968, 0.7124550, -1.0858630, 1.0139599
1: -0.5986170, 0.7400016, -0.8372884, 1.0017992, -1.6004162, 1.5772898
2: -0.4129052, 0.7700409, -0.5536113, 1.1024923, -1.5153975, 1.3236523
3: -0.9442486, 0.9823394, -1.3855500, 1.3398813, -2.2841299, 2.3678887
4: -0.5904477, 1.0167003, -0.8118663, 1.4754035, -2.0658512, 1.8285666

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3053402, upper bound: 3.2896934
time: 0.43 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3127344, upper bound: 3.3052940
time: 0.44 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3127344, upper bound: 3.3052940
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.3100591, 0.4353619, -0.1803700, 0.1815150, -0.4915742, 0.6157318
1: -0.5008603, 0.6214672, -0.2514807, 0.2583605, -0.7592206, 0.8729479
2: -0.3562234, 0.6509233, -0.2205296, 0.2771387, -0.6333621, 0.8714529
3: -0.7530009, 0.8102221, -0.3344405, 0.3125712, -1.0655720, 1.1446626
4: -0.4713841, 0.8378688, -0.2086931, 0.3468166, -0.8182006, 1.0465620

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2024730, upper bound: 3.3147914
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2024730, upper bound: 3.3147914
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.3320688, 0.4660973, -0.3721623, 0.5198109, -0.8518795, 0.8382596
1: -0.5381686, 0.6677936, -0.6154165, 0.7582794, -1.2964478, 1.2832100
2: -0.3777248, 0.6980434, -0.4279845, 0.7296137, -1.1073382, 1.1260279
3: -0.8200477, 0.8725698, -0.8802427, 1.0171561, -1.8372039, 1.7528125
4: -0.5075729, 0.9035305, -0.6110348, 0.9500679, -1.4576406, 1.5145652

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2850447, upper bound: 3.3538933
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2850447, upper bound: 3.3648028
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.5475030, 0.8051696, -0.1752371, 0.1600563, -0.7075592, 0.9804068
1: -0.9047095, 1.1165475, -0.2410980, 0.2296271, -1.1343366, 1.3576455
2: -0.5974170, 1.2467538, -0.2133930, 0.2479965, -0.8454133, 1.4601468
3: -1.5290978, 1.4838481, -0.3201365, 0.2728859, -1.8019837, 1.8039846
4: -0.8865705, 1.6566507, -0.1963671, 0.3093478, -1.1959183, 1.8530178

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 17

Time for candidate selection: 4.37 seconds

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1805324, upper bound: 3.2347830
time: 0.46 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1254930, upper bound: 3.2127386
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 34

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 17

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 5
type: A, layer: 5, pos: 44
type: A, layer: 5, pos: 11
type: A, layer: 5, pos: 27
type: A, layer: 5, pos: 5
type: A, layer: 5, pos: 49
type: A, layer: 5, pos: 37
type: A, layer: 5, pos: 8
type: A, layer: 5, pos: 45
type: A, layer: 5, pos: 42
type: A, layer: 5, pos: 24
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 20
type: A, layer: 5, pos: 38
type: A, layer: 5, pos: 14
type: A, layer: 5, pos: 4
type: A, layer: 5, pos: 31
type: A, layer: 5, pos: 15

Time for candidate selection: 10.20 seconds

### Candidate
type: A, layer: 5, pos: 44

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1742380, upper bound: 3.2316537
time: 0.45 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1717436, upper bound: 3.2253274
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.5010968, 0.7124550, -0.3437773, 0.4621157, -0.9632120, 1.0562323
1: -0.8372884, 1.0017992, -0.5606074, 0.6905864, -1.5278747, 1.5624065
2: -0.5536113, 1.1024923, -0.3961424, 0.6358137, -1.1894250, 1.4986348
3: -1.3855500, 1.3398813, -0.7636503, 0.9227027, -2.3082523, 2.1035316
4: -0.8118663, 1.4754035, -0.5569270, 0.8112569, -1.6231233, 2.0323305

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2956179, upper bound: 3.3127344
time: 0.49 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3052940, upper bound: 3.3127081
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.3753012, 0.5288189, -0.3753012, 0.5288189, -0.9041201, 0.9041200
1: -0.6084703, 0.7558654, -0.6084703, 0.7558654, -1.3643357, 1.3643357
2: -0.4198778, 0.7994800, -0.4198778, 0.7994800, -1.2193577, 1.2193577
3: -0.9523419, 0.9947283, -0.9523419, 0.9947283, -1.9470696, 1.9470699
4: -0.5820951, 1.0479591, -0.5820951, 1.0479591, -1.6300540, 1.6300540

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2816323, upper bound: 3.3429504
time: 0.48 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3303154, upper bound: 3.3674660
time: 0.44 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.3753012, 0.5288189, -0.6188753, 0.9653776, -1.3406787, 1.1476941
1: -0.6084703, 0.7558654, -1.0178568, 1.3381594, -1.9466296, 1.7737222
2: -0.4198778, 0.7994800, -0.6695868, 1.4706283, -1.8905060, 1.4690667
3: -0.9523419, 0.9947283, -1.7465825, 1.7551888, -2.7075305, 2.7413108
4: -0.5820951, 1.0479591, -1.0122970, 1.9336305, -2.5157256, 2.0602558

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2816323, upper bound: 3.3429504
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3303154, upper bound: 3.3674660
time: 0.44 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.6188753, 0.9653776, -0.3753012, 0.5288189, -1.1476941, 1.3406787
1: -1.0178568, 1.3381594, -0.6084703, 0.7558654, -1.7737221, 1.9466296
2: -0.6695868, 1.4706283, -0.4198778, 0.7994800, -1.4690667, 1.8905060
3: -1.7465825, 1.7551888, -0.9523419, 0.9947283, -2.7413106, 2.7075305
4: -1.0122970, 1.9336305, -0.5820951, 1.0479591, -2.0602555, 2.5157256

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3134799, upper bound: 3.3134014
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3135253, upper bound: 3.3135253
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.6188753, 0.9653776, -0.6188753, 0.9653776, -1.5842528, 1.5842528
1: -1.0178568, 1.3381594, -1.0178568, 1.3381594, -2.3560159, 2.3560159
2: -0.6695868, 1.4706283, -0.6695868, 1.4706283, -2.1402152, 2.1402152
3: -1.7465825, 1.7551888, -1.7465825, 1.7551888, -3.5017715, 3.5017715
4: -1.0122970, 1.9336305, -1.0122970, 1.9336305, -2.9459276, 2.9459276

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3134799, upper bound: 3.3134014
time: 0.43 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3135253, upper bound: 3.3135253
time: 0.46 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.23 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 0, lower bound: -3.1870526, upper bound: 3.2741617
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 0, lower bound: -3.1566236, upper bound: 3.1566236
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 0, lower bound: -3.1870526, upper bound: 3.2987063
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 0, lower bound: -3.1566236, upper bound: 3.1816649
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 0, lower bound: -3.3087832, upper bound: 3.3054649
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 0, lower bound: -3.3098656, upper bound: 3.3098627
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 0, lower bound: -3.3087832, upper bound: 3.3054678
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 0, lower bound: -3.3098656, upper bound: 3.3098627
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 0, lower bound: -3.2347830, upper bound: 3.1805324
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 0, lower bound: -3.2347830, upper bound: 3.1805324
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 0, lower bound: -3.2347830, upper bound: 3.1805324
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 0, lower bound: -3.2347830, upper bound: 3.1805324
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 0, lower bound: -3.3127344, upper bound: 3.3052940
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 0, lower bound: -3.3127344, upper bound: 3.3052940
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 0, lower bound: -3.3127344, upper bound: 3.3052940
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 0, lower bound: -3.3127344, upper bound: 3.3052940
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 0, lower bound: -3.2024730, upper bound: 3.3147914
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 0, lower bound: -3.2024730, upper bound: 3.3147914
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 0, lower bound: -3.2850447, upper bound: 3.3538933
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 0, lower bound: -3.2850447, upper bound: 3.3648028
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 0, lower bound: -3.1742380, upper bound: 3.2316537
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 0, lower bound: -3.1717436, upper bound: 3.2253274
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 0, lower bound: -3.2956179, upper bound: 3.3127344
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 0, lower bound: -3.3052940, upper bound: 3.3127081
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 0, lower bound: -3.2816323, upper bound: 3.3429504
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 0, lower bound: -3.3303154, upper bound: 3.3674660
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 0, lower bound: -3.2816323, upper bound: 3.3429504
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 0, lower bound: -3.3303154, upper bound: 3.3674660
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 0, lower bound: -3.3134799, upper bound: 3.3134014
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 0, lower bound: -3.3135253, upper bound: 3.3135253
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 0, lower bound: -3.3134799, upper bound: 3.3134014
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 0, lower bound: -3.3135253, upper bound: 3.3135253

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0676768, 0.0203616, -0.1803700, 0.1815150, -0.2491919, 0.2007315
1: -0.0773866, 0.0391565, -0.2514807, 0.2583605, -0.3357471, 0.2906372
2: -0.0644345, 0.0282484, -0.2205296, 0.2771387, -0.3415731, 0.2487780
3: -0.0712000, 0.0471544, -0.3344405, 0.3125712, -0.3837712, 0.3815949
4: -0.0532835, 0.0354563, -0.2086931, 0.3468166, -0.4001001, 0.2441494

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1740394, upper bound: 3.2409801
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0696478, upper bound: 3.1655834
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1870526, upper bound: 3.2741617
time: 0.39 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.1649871, 0.1174907, -0.1752371, 0.1600563, -0.3250434, 0.2927279
1: -0.2215959, 0.1765458, -0.2410980, 0.2296271, -0.4512230, 0.4176438
2: -0.1996021, 0.1877475, -0.2133930, 0.2479965, -0.4475986, 0.4011405
3: -0.2914082, 0.2083298, -0.3201365, 0.2728859, -0.5642942, 0.5284663
4: -0.1816220, 0.2294925, -0.1963671, 0.3093478, -0.4909698, 0.4258595

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1566236, upper bound: 3.1566236
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1566236, upper bound: 3.1566236
time: 0.40 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0676768, 0.0203616, -0.3707971, 0.5178647, -0.5855415, 0.3911586
1: -0.0773866, 0.0391565, -0.6134025, 0.7549255, -0.8323121, 0.6525590
2: -0.0644345, 0.0282484, -0.4267291, 0.7276129, -0.7920474, 0.4549775
3: -0.0712000, 0.0471544, -0.8775759, 1.0125093, -1.0837094, 0.9247303
4: -0.0532835, 0.0354563, -0.6082076, 0.9479311, -1.0012146, 0.6436639

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2523287, upper bound: 3.2918128
time: 0.36 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2383201, upper bound: 3.1816649
time: 0.51 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2383201, upper bound: 3.1816649
time: 0.41 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.1649871, 0.1174907, -0.3437773, 0.4752336, -0.6402207, 0.4612680
1: -0.2215959, 0.1765458, -0.5637994, 0.6905864, -0.9121822, 0.7403452
2: -0.1996021, 0.1877475, -0.3961424, 0.6675218, -0.8671240, 0.5838899
3: -0.2914082, 0.2083298, -0.8024915, 0.9227027, -1.2141109, 1.0108213
4: -0.1816220, 0.2294925, -0.5569270, 0.8650054, -1.0466274, 0.7864194

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2383201, upper bound: 3.1816649
time: 0.43 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2383201, upper bound: 3.1816649
time: 0.40 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.2559387, 0.3415756, -0.1912085, 0.2109509, -0.4668896, 0.5327840
1: -0.3946697, 0.4863652, -0.2706243, 0.2982620, -0.6929317, 0.7569895
2: -0.2981569, 0.5062003, -0.2345207, 0.3197280, -0.6178848, 0.7407210
3: -0.5855743, 0.6284050, -0.3691734, 0.3672814, -0.9528557, 0.9975784
4: -0.3760676, 0.6415557, -0.2326370, 0.4024982, -0.7785658, 0.8741927

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2918128, upper bound: 3.2523287
time: 0.43 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3055809, upper bound: 3.3043322
time: 0.38 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3087832, upper bound: 3.3054649
time: 0.41 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.4087062, 0.5647650, -0.1721811, 0.1581199, -0.5668261, 0.7369461
1: -0.6539204, 0.7966581, -0.2369932, 0.2269053, -0.8808258, 1.0336512
2: -0.4457314, 0.8629469, -0.2096886, 0.2443709, -0.6901023, 1.0726355
3: -1.0570911, 1.0608249, -0.3134334, 0.2714462, -1.3285372, 1.3742583
4: -0.6360155, 1.1408229, -0.1938250, 0.3040901, -0.9401056, 1.3346479

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3067020, upper bound: 3.3087506
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3098656, upper bound: 3.3098627
time: 0.41 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.2559387, 0.3415756, -0.4002510, 0.5594531, -0.8153918, 0.7418266
1: -0.3946697, 0.4863652, -0.6648765, 0.8165386, -1.2112083, 1.1512417
2: -0.2981569, 0.5062003, -0.4584243, 0.7870062, -1.0851631, 0.9646246
3: -0.5855743, 0.6284050, -0.9596827, 1.0986850, -1.6842594, 1.5880877
4: -0.3760676, 0.6415557, -0.6609793, 1.0315057, -1.4075732, 1.3025349

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3043861, upper bound: 3.3043658
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3043861, upper bound: 3.3054678
time: 0.45 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.4087062, 0.5647650, -0.3491561, 0.4875938, -0.8963000, 0.9139211
1: -0.6539204, 0.7966581, -0.5758667, 0.7097453, -1.3636658, 1.3725247
2: -0.4457314, 0.8629469, -0.4033471, 0.6837229, -1.1294543, 1.2662940
3: -1.0570911, 1.0608249, -0.8177493, 0.9496778, -2.0067687, 1.8785741
4: -0.6360155, 1.1408229, -0.5694224, 0.8863442, -1.5223597, 1.7102454

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3053682, upper bound: 3.3086207
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3053682, upper bound: 3.3098627
time: 0.46 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0676768, 0.0203616, -0.3100591, 0.4353619, -0.5030388, 0.3304206
1: -0.0773866, 0.0391565, -0.5008603, 0.6214672, -0.6988539, 0.5400168
2: -0.0644345, 0.0282484, -0.3562234, 0.6509233, -0.7153578, 0.3844718
3: -0.0712000, 0.0471544, -0.7530009, 0.8102221, -0.8814221, 0.8001552
4: -0.0532835, 0.0354563, -0.4713841, 0.8378688, -0.8911523, 0.5068403

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3147914, upper bound: 3.2024730
time: 0.43 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3147914, upper bound: 3.2024730
time: 0.43 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.1649871, 0.1174907, -0.3100591, 0.4353619, -0.6003490, 0.4275497
1: -0.2215959, 0.1765458, -0.5008603, 0.6214672, -0.8430631, 0.6774061
2: -0.1996021, 0.1877475, -0.3562234, 0.6509233, -0.8505254, 0.5439708
3: -0.2914082, 0.2083298, -0.7530009, 0.8102221, -1.1016303, 0.9613307
4: -0.1816220, 0.2294925, -0.4713841, 0.8378688, -1.0194908, 0.7008766

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3147914, upper bound: 3.2024730
time: 0.45 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3147914, upper bound: 3.2024730
time: 0.43 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0676768, 0.0203616, -0.5475030, 0.8051696, -0.8728464, 0.5678644
1: -0.0773866, 0.0391565, -0.9047095, 1.1165475, -1.1939341, 0.9438660
2: -0.0644345, 0.0282484, -0.5974170, 1.2467538, -1.3111883, 0.6256654
3: -0.0712000, 0.0471544, -1.5290978, 1.4838481, -1.5550481, 1.5762522
4: -0.0532835, 0.0354563, -0.8865705, 1.6566507, -1.7099342, 0.9220268

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 21
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 17

Time for candidate selection: 4.48 seconds

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2347830, upper bound: 3.1805324
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 21

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 34

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 17

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 5
type: B, layer: 5, pos: 44
type: B, layer: 5, pos: 11
type: B, layer: 5, pos: 27
type: B, layer: 5, pos: 5
type: B, layer: 5, pos: 49
type: B, layer: 5, pos: 37
type: B, layer: 5, pos: 8
type: B, layer: 5, pos: 45
type: B, layer: 5, pos: 42
type: B, layer: 5, pos: 24
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 20
type: B, layer: 5, pos: 38
type: B, layer: 5, pos: 14
type: B, layer: 5, pos: 4
type: B, layer: 5, pos: 31
type: B, layer: 5, pos: 15

Time for candidate selection: 9.22 seconds

### Candidate
type: B, layer: 5, pos: 44

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2316537, upper bound: 3.1742380
time: 0.43 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2253274, upper bound: 3.1717436
time: 0.44 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.1649871, 0.1174907, -0.5475030, 0.8051696, -0.9701567, 0.6649937
1: -0.2215959, 0.1765458, -0.9047095, 1.1165475, -1.3381433, 1.0812552
2: -0.1996021, 0.1877475, -0.5974170, 1.2467538, -1.4463559, 0.7851645
3: -0.2914082, 0.2083298, -1.5290978, 1.4838481, -1.7752563, 1.7374276
4: -0.1816220, 0.2294925, -0.8865705, 1.6566507, -1.8382727, 1.1160630

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 21
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 17

Time for candidate selection: 4.53 seconds

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2347830, upper bound: 3.1805324
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 21

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 34

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 17

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 5
type: B, layer: 5, pos: 44
type: B, layer: 5, pos: 11
type: B, layer: 5, pos: 27
type: B, layer: 5, pos: 5
type: B, layer: 5, pos: 49
type: B, layer: 5, pos: 37
type: B, layer: 5, pos: 8
type: B, layer: 5, pos: 45
type: B, layer: 5, pos: 42
type: B, layer: 5, pos: 24
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 20
type: B, layer: 5, pos: 38
type: B, layer: 5, pos: 14
type: B, layer: 5, pos: 4
type: B, layer: 5, pos: 31
type: B, layer: 5, pos: 15

Time for candidate selection: 9.28 seconds

### Candidate
type: B, layer: 5, pos: 44

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2316537, upper bound: 3.1742380
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2253274, upper bound: 3.1717436
time: 0.43 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.1850611, 0.2096545, -0.3320688, 0.4660973, -0.6511584, 0.5417233
1: -0.2660490, 0.2913542, -0.5381686, 0.6677936, -0.9338427, 0.8295228
2: -0.2290718, 0.3008571, -0.3777248, 0.6980434, -0.9271152, 0.6785818
3: -0.3428191, 0.3601812, -0.8200477, 0.8725698, -1.2153889, 1.1802289
4: -0.2267372, 0.3690930, -0.5075729, 0.9035305, -1.1302677, 0.8766658

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3538933, upper bound: 3.2850447
time: 0.46 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3538933, upper bound: 3.3189540
time: 0.45 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.3088257, 0.4204044, -0.3320688, 0.4660973, -0.7749230, 0.7524731
1: -0.4969721, 0.6138726, -0.5381686, 0.6677936, -1.1647657, 1.1520412
2: -0.3497591, 0.6176917, -0.3777248, 0.6980434, -1.0478024, 0.9954162
3: -0.7490399, 0.8086340, -0.8200477, 0.8725698, -1.6216097, 1.6286815
4: -0.4839897, 0.8007323, -0.5075729, 0.9035305, -1.3875202, 1.3083051

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3538933, upper bound: 3.2850447
time: 0.43 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3538933, upper bound: 3.3189540
time: 0.44 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.1850611, 0.2096545, -0.5010968, 0.7124550, -0.8975161, 0.7107512
1: -0.2660490, 0.2913542, -0.8372884, 1.0017992, -1.2678483, 1.1286426
2: -0.2290718, 0.3008571, -0.5536113, 1.1024923, -1.3315642, 0.8544684
3: -0.3428191, 0.3601812, -1.3855500, 1.3398813, -1.6827004, 1.7457312
4: -0.2267372, 0.3690930, -0.8118663, 1.4754035, -1.7021408, 1.1809592

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3127344, upper bound: 3.2956179
time: 0.45 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3127081, upper bound: 3.3052940
time: 0.53 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.3088257, 0.4204044, -0.5010968, 0.7124550, -1.0212806, 0.9215012
1: -0.4969721, 0.6138726, -0.8372884, 1.0017992, -1.4987713, 1.4511610
2: -0.3497591, 0.6176917, -0.5536113, 1.1024923, -1.4522514, 1.1713030
3: -0.7490399, 0.8086340, -1.3855500, 1.3398813, -2.0889211, 2.1941836
4: -0.4839897, 0.8007323, -0.8118663, 1.4754035, -1.9593933, 1.6125987

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3127344, upper bound: 3.2956179
time: 0.47 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3127081, upper bound: 3.3052940
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.2211066, 0.2775607, -0.1803700, 0.1815150, -0.4026217, 0.4579307
1: -0.3221700, 0.3887413, -0.2514807, 0.2583605, -0.5805305, 0.6402220
2: -0.2730529, 0.4024681, -0.2205296, 0.2771387, -0.5501916, 0.6229978
3: -0.4342754, 0.4909782, -0.3344405, 0.3125712, -0.7468467, 0.8254187
4: -0.2787128, 0.4985040, -0.2086931, 0.3468166, -0.6255294, 0.7071971

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1901709, upper bound: 3.2980723
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1973792, upper bound: 3.3099443
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2024730, upper bound: 3.3147914
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2024730, upper bound: 3.3147914
time: 0.45 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.6112540, 0.8508660, -0.1803700, 0.1815150, -0.7927690, 1.0312359
1: -0.9703345, 1.2209729, -0.2514807, 0.2583605, -1.2286947, 1.4724536
2: -0.6424070, 1.3305063, -0.2205296, 0.2771387, -0.9195457, 1.5510360
3: -1.6326315, 1.6337559, -0.3344405, 0.3125712, -1.9452028, 1.9681964
4: -0.9662173, 1.7874435, -0.2086931, 0.3468166, -1.3130338, 1.9961367

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1901709, upper bound: 3.2980723
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1973792, upper bound: 3.3099443
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2024730, upper bound: 3.3147914
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2024730, upper bound: 3.3147914
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.2211066, 0.2775607, -0.3721623, 0.5198109, -0.7409176, 0.6497231
1: -0.3221700, 0.3887413, -0.6154165, 0.7582794, -1.0804495, 1.0041578
2: -0.2730529, 0.4024681, -0.4279845, 0.7296137, -1.0026666, 0.8304526
3: -0.4342754, 0.4909782, -0.8802427, 1.0171561, -1.4514315, 1.3712208
4: -0.2787128, 0.4985040, -0.6110348, 0.9500679, -1.2287807, 1.1095388

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1973792, upper bound: 3.3490080
time: 0.46 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2024730, upper bound: 3.3538933
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2024730, upper bound: 3.3147914
time: 0.44 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.6227801, 0.8648329, -0.3721623, 0.5198109, -1.1425909, 1.2369952
1: -0.9949624, 1.2503651, -0.6154165, 0.7582794, -1.7532418, 1.8657815
2: -0.6539169, 1.3625121, -0.4279845, 0.7296137, -1.3835304, 1.7904966
3: -1.6873388, 1.6740556, -0.8802427, 1.0171561, -2.7044950, 2.5542984
4: -0.9924154, 1.8411534, -0.6110348, 0.9500679, -1.9424834, 2.4521880

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1973792, upper bound: 3.3464861
time: 0.45 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2024730, upper bound: 3.3562831
time: 0.44 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2024730, upper bound: 3.3562831
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.4130764, 0.5761445, -0.1752371, 0.1600563, -0.5731326, 0.7513816
1: -0.6916107, 0.8024566, -0.2410980, 0.2296271, -0.9212378, 1.0435547
2: -0.4589225, 0.8798022, -0.2133930, 0.2479965, -0.7069190, 1.0931952
3: -1.1023782, 1.0841615, -0.3201365, 0.2728859, -1.3752642, 1.4042981
4: -0.6645420, 1.1869493, -0.1963671, 0.3093478, -0.9738898, 1.3833163

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1742380, upper bound: 3.2316537
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1742380, upper bound: 3.2316537
time: 0.49 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -2.3698783, 4.5716200, -0.1752371, 0.1600563, -2.5299344, 4.7468572
1: -3.6489408, 6.1731424, -0.2410980, 0.2296271, -3.8785679, 6.4142404
2: -2.3092916, 6.6463075, -0.2133930, 0.2479965, -2.5572882, 6.8597007
3: -6.8561640, 7.6015148, -0.3201365, 0.2728859, -7.1290493, 7.9216514
4: -3.7684946, 8.4313126, -0.1963671, 0.3093478, -4.0778422, 8.6276798

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1717436, upper bound: 3.2253274
time: 0.45 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1717436, upper bound: 3.2253274
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.5771171, 0.8609411, -0.3185214, 0.4298684, -1.0069855, 1.1794624
1: -0.9539698, 1.2141869, -0.5172459, 0.6381128, -1.5920826, 1.7314329
2: -0.6286759, 1.3290529, -0.3689098, 0.5933194, -1.2219954, 1.6979628
3: -1.6407450, 1.6043134, -0.7057850, 0.8490622, -2.4898064, 2.3100984
4: -0.9467582, 1.7622975, -0.5116025, 0.7551441, -1.7019019, 2.2739000

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2881534, upper bound: 3.3053401
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2956179, upper bound: 3.3127344
time: 0.45 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2956179, upper bound: 3.3127344
time: 0.45 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.3838726, 0.5261831, -0.3437773, 0.4621157, -0.8459882, 0.8699604
1: -0.6385440, 0.7297446, -0.5606074, 0.6905864, -1.3291304, 1.2903519
2: -0.4337534, 0.7962208, -0.3961424, 0.6358137, -1.0695671, 1.1923633
3: -1.0099097, 0.9724351, -0.7636503, 0.9227027, -1.9326122, 1.7360854
4: -0.5983632, 1.0662262, -0.5569270, 0.8112569, -1.4096202, 1.6231532

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2896934, upper bound: 3.3046225
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3052940, upper bound: 3.3127081
time: 0.44 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3052940, upper bound: 3.3127081
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.2211066, 0.2775607, -0.3100591, 0.4353619, -0.6564685, 0.5876198
1: -0.3221700, 0.3887413, -0.5008603, 0.6214672, -0.9436373, 0.8896016
2: -0.2730529, 0.4024681, -0.3562234, 0.6509233, -0.9239762, 0.7586914
3: -0.4342754, 0.4909782, -0.7530009, 0.8102221, -1.2444975, 1.2439790
4: -0.2787128, 0.4985040, -0.4713841, 0.8378688, -1.1165817, 0.9698879

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3922332, upper bound: 3.3922332
time: 0.49 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3922332, upper bound: 3.3922332
time: 0.45 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.6227801, 0.8648329, -0.3320688, 0.4660973, -1.0888774, 1.1969016
1: -0.9949624, 1.2503651, -0.5381686, 0.6677936, -1.6627556, 1.7885332
2: -0.6539169, 1.3625121, -0.3777248, 0.6980434, -1.3519602, 1.7402369
3: -1.6873388, 1.6740556, -0.8200477, 0.8725698, -2.5599086, 2.4941034
4: -0.9924154, 1.8411534, -0.5075729, 0.9035305, -1.8959459, 2.3487263

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3922332, upper bound: 3.3922332
time: 0.45 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3922332, upper bound: 3.3922332
time: 0.47 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.2211066, 0.2775607, -0.5475030, 0.8051696, -1.0262762, 0.8250637
1: -0.3221700, 0.3887413, -0.9047095, 1.1165475, -1.4387175, 1.2934507
2: -0.2730529, 0.4024681, -0.5974170, 1.2467538, -1.5198067, 0.9998850
3: -0.4342754, 0.4909782, -1.5290978, 1.4838481, -1.9181235, 2.0200753
4: -0.2787128, 0.4985040, -0.8865705, 1.6566507, -1.9353635, 1.3850745

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2760649, upper bound: 3.3389726
time: 0.46 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2715491, upper bound: 3.3301218
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 21
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 17

Time for candidate selection: 5.52 seconds

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 21

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2575306, upper bound: 3.3262513
time: 0.46 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2558481, upper bound: 3.3260753
time: 0.49 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.6227801, 0.8648329, -0.5010968, 0.7124550, -1.3352351, 1.3659291
1: -0.9949624, 1.2503651, -0.8372884, 1.0017992, -1.9967616, 2.0876536
2: -0.6539169, 1.3625121, -0.5536113, 1.1024923, -1.7564090, 1.9161234
3: -1.6873388, 1.6740556, -1.3855500, 1.3398813, -3.0272202, 3.0596049
4: -0.9924154, 1.8411534, -0.8118663, 1.4754035, -2.4678190, 2.6530194

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3235557, upper bound: 3.3326271
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3237142, upper bound: 3.3378274
time: 0.46 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.4467238, 0.6215951, -0.3753012, 0.5288189, -0.9755427, 0.9968962
1: -0.7328915, 0.8427354, -0.6084703, 0.7558654, -1.4887569, 1.4512056
2: -0.4924225, 0.9656458, -0.4198778, 0.7994800, -1.2919024, 1.3855237
3: -1.2085794, 1.1289421, -0.9523419, 0.9947283, -2.2033076, 2.0812838
4: -0.7015048, 1.3040695, -0.5820951, 1.0479591, -1.7494637, 1.8861645

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3391176, upper bound: 3.3194033
time: 0.45 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3228557, upper bound: 3.3156158
time: 0.46 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.4117214, 0.5673239, -0.2834730, 0.3991817, -0.8109032, 0.8507968
1: -0.6853573, 0.7768623, -0.4585487, 0.5693799, -1.2547371, 1.2354106
2: -0.4597958, 0.8666953, -0.3304136, 0.5910300, -1.0508258, 1.1971086
3: -1.0987918, 1.0422142, -0.6734245, 0.7400450, -1.8388366, 1.7156386
4: -0.6412443, 1.1624334, -0.4287502, 0.7552848, -1.3965291, 1.5911837

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3290509, upper bound: 3.3159920
time: 0.45 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3212138, upper bound: 3.3140238
time: 0.44 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.4467238, 0.6215951, -0.6188753, 0.9653776, -1.4121013, 1.2404703
1: -0.7328915, 0.8427354, -1.0178568, 1.3381594, -2.0710504, 1.8605920
2: -0.4924225, 0.9656458, -0.6695868, 1.4706283, -1.9630504, 1.6352326
3: -1.2085794, 1.1289421, -1.7465825, 1.7551888, -2.9637682, 2.8755245
4: -0.7015048, 1.3040695, -1.0122970, 1.9336305, -2.6351352, 2.3163664

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3132533, upper bound: 3.3132533
time: 0.43 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3132533, upper bound: 3.3133037
time: 0.46 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.4117214, 0.5673239, -0.5303482, 0.7649302, -1.1766517, 1.0976720
1: -0.6853573, 0.7768623, -0.8779442, 1.0630480, -1.7484052, 1.6548064
2: -0.4597958, 0.8666953, -0.5792612, 1.1890204, -1.6488162, 1.4459562
3: -1.0987918, 1.0422142, -1.4742205, 1.4190956, -2.5178876, 2.5164347
4: -0.6412443, 1.1624334, -0.8569046, 1.5868173, -2.2280617, 2.0193379

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3132705, upper bound: 3.3125449
time: 0.45 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3122901, upper bound: 3.3122901
time: 0.44 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.36 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 0, lower bound: -3.0696478, upper bound: 3.1655834
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 0, lower bound: -3.1870526, upper bound: 3.2741617
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 0, lower bound: -3.1566236, upper bound: 3.1566236
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 0, lower bound: -3.1566236, upper bound: 3.1566236
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 0, lower bound: -3.2383201, upper bound: 3.1816649
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 0, lower bound: -3.2383201, upper bound: 3.1816649
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 0, lower bound: -3.2383201, upper bound: 3.1816649
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 0, lower bound: -3.2383201, upper bound: 3.1816649
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 0, lower bound: -3.3055809, upper bound: 3.3043322
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 0, lower bound: -3.3087832, upper bound: 3.3054649
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 0, lower bound: -3.3067020, upper bound: 3.3087506
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 0, lower bound: -3.3098656, upper bound: 3.3098627
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 0, lower bound: -3.3043861, upper bound: 3.3043658
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 0, lower bound: -3.3043861, upper bound: 3.3054678
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 0, lower bound: -3.3053682, upper bound: 3.3086207
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 0, lower bound: -3.3053682, upper bound: 3.3098627
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 0, lower bound: -3.3147914, upper bound: 3.2024730
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 0, lower bound: -3.3147914, upper bound: 3.2024730
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 0, lower bound: -3.3147914, upper bound: 3.2024730
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 0, lower bound: -3.3147914, upper bound: 3.2024730
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 0, lower bound: -3.2316537, upper bound: 3.1742380
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 0, lower bound: -3.2253274, upper bound: 3.1717436
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 0, lower bound: -3.2316537, upper bound: 3.1742380
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 0, lower bound: -3.2253274, upper bound: 3.1717436
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 0, lower bound: -3.3538933, upper bound: 3.2850447
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 0, lower bound: -3.3538933, upper bound: 3.3189540
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 0, lower bound: -3.3538933, upper bound: 3.2850447
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 0, lower bound: -3.3538933, upper bound: 3.3189540
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 0, lower bound: -3.3127344, upper bound: 3.2956179
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 0, lower bound: -3.3127081, upper bound: 3.3052940
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 0, lower bound: -3.3127344, upper bound: 3.2956179
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 0, lower bound: -3.3127081, upper bound: 3.3052940
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 0, lower bound: -3.2024730, upper bound: 3.3147914
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 0, lower bound: -3.2024730, upper bound: 3.3147914
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 0, lower bound: -3.2024730, upper bound: 3.3147914
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 0, lower bound: -3.2024730, upper bound: 3.3147914
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 0, lower bound: -3.2024730, upper bound: 3.3538933
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 0, lower bound: -3.2024730, upper bound: 3.3147914
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 0, lower bound: -3.2024730, upper bound: 3.3562831
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 0, lower bound: -3.2024730, upper bound: 3.3562831
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 0, lower bound: -3.1742380, upper bound: 3.2316537
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 0, lower bound: -3.1742380, upper bound: 3.2316537
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 0, lower bound: -3.1717436, upper bound: 3.2253274
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 0, lower bound: -3.1717436, upper bound: 3.2253274
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 0, lower bound: -3.2956179, upper bound: 3.3127344
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 0, lower bound: -3.2956179, upper bound: 3.3127344
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 0, lower bound: -3.3052940, upper bound: 3.3127081
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 0, lower bound: -3.3052940, upper bound: 3.3127081
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 0, lower bound: -3.3922332, upper bound: 3.3922332
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 0, lower bound: -3.3922332, upper bound: 3.3922332
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 0, lower bound: -3.3922332, upper bound: 3.3922332
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 0, lower bound: -3.3922332, upper bound: 3.3922332
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 0, lower bound: -3.2575306, upper bound: 3.3262513
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 0, lower bound: -3.2558481, upper bound: 3.3260753
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 0, lower bound: -3.3235557, upper bound: 3.3326271
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 0, lower bound: -3.3237142, upper bound: 3.3378274
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 0, lower bound: -3.3391176, upper bound: 3.3194033
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 0, lower bound: -3.3228557, upper bound: 3.3156158
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 0, lower bound: -3.3290509, upper bound: 3.3159920
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 0, lower bound: -3.3212138, upper bound: 3.3140238
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 0, lower bound: -3.3132533, upper bound: 3.3132533
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 0, lower bound: -3.3132533, upper bound: 3.3133037
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 0, lower bound: -3.3132705, upper bound: 3.3125449
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.36
Output dim: 0, lower bound: -3.3122901, upper bound: 3.3122901

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0652687, 0.0085601, -0.1519704, 0.0854497, -0.1507184, 0.1605306
1: -0.0735153, 0.0199839, -0.1990021, 0.1323870, -0.2059023, 0.2189861
2: -0.0615145, 0.0108451, -0.1821267, 0.1375749, -0.1990893, 0.1929718
3: -0.0668982, 0.0275343, -0.2569412, 0.1586734, -0.2255716, 0.2844754
4: -0.0480196, 0.0146193, -0.1674782, 0.1618953, -0.2099149, 0.1820975

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0694806, upper bound: 3.1649246
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0694806, upper bound: 3.1655834
time: 0.40 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0676768, 0.0203616, -0.1717433, 0.1517706, -0.2194474, 0.1921049
1: -0.0773866, 0.0391565, -0.2358323, 0.2205657, -0.2979524, 0.2749888
2: -0.0644345, 0.0282484, -0.2091871, 0.2349725, -0.2994070, 0.2374355
3: -0.0712000, 0.0471544, -0.3101773, 0.2635337, -0.3347336, 0.3573318
4: -0.0532835, 0.0354563, -0.1933890, 0.2911050, -0.3443885, 0.2288453

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9203093, upper bound: 3.0133196
time: 0.43 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1734842, upper bound: 3.2703347
time: 0.44 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1734842, upper bound: 3.2741617
time: 0.41 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.1649871, 0.1174907, -0.0676768, 0.0203616, -0.1853487, 0.1851676
1: -0.2215959, 0.1765458, -0.0773866, 0.0391565, -0.2607524, 0.2539324
2: -0.1996021, 0.1877475, -0.0644345, 0.0282484, -0.2278505, 0.2521820
3: -0.2914082, 0.2083298, -0.0712000, 0.0471544, -0.3385626, 0.2795298
4: -0.1816220, 0.2294925, -0.0532835, 0.0354563, -0.2170783, 0.2827760

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0994107, upper bound: 3.0483785
time: 0.44 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1566236, upper bound: 3.1566236
time: 0.40 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.1649871, 0.1174907, -0.1649871, 0.1174907, -0.2824779, 0.2824779
1: -0.2215959, 0.1765458, -0.2215959, 0.1765458, -0.3981417, 0.3981417
2: -0.1996021, 0.1877475, -0.1996021, 0.1877475, -0.3873496, 0.3873496
3: -0.2914082, 0.2083298, -0.2914082, 0.2083298, -0.4997380, 0.4997380
4: -0.1816220, 0.2294925, -0.1816220, 0.2294925, -0.4111145, 0.4111145

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0994107, upper bound: 3.0483785
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1566236, upper bound: 3.1566236
time: 0.40 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0676768, 0.0203616, -0.1842154, 0.2073062, -0.2749831, 0.2045770
1: -0.0773866, 0.0391565, -0.2644064, 0.2852268, -0.3626135, 0.3035629
2: -0.0644345, 0.0282484, -0.2280039, 0.2978324, -0.3622669, 0.2562523
3: -0.0712000, 0.0471544, -0.3391942, 0.3525377, -0.4237376, 0.3863487
4: -0.0532835, 0.0354563, -0.2241528, 0.3658484, -0.4191319, 0.2596091

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1757587, upper bound: 3.2713787
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2657301, upper bound: 3.2987063
time: 0.47 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0676768, 0.0203616, -0.2918625, 0.3971892, -0.4648660, 0.3122240
1: -0.0773866, 0.0391565, -0.4734699, 0.5811095, -0.6584961, 0.5126265
2: -0.0644345, 0.0282484, -0.3393690, 0.5553786, -0.6198131, 0.3676174
3: -0.0712000, 0.0471544, -0.6585814, 0.7684231, -0.8396230, 0.7057359
4: -0.0532835, 0.0354563, -0.4620743, 0.7093326, -0.7626161, 0.4975306

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1757587, upper bound: 3.2713787
time: 0.45 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2657301, upper bound: 3.2987063
time: 0.46 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.1649871, 0.1174907, -0.1842154, 0.2073062, -0.3722934, 0.3017061
1: -0.2215959, 0.1765458, -0.2644064, 0.2852268, -0.5068227, 0.4409522
2: -0.1996021, 0.1877475, -0.2280039, 0.2978324, -0.4974345, 0.4157514
3: -0.2914082, 0.2083298, -0.3391942, 0.3525377, -0.6439459, 0.5475241
4: -0.1816220, 0.2294925, -0.2241528, 0.3658484, -0.5474705, 0.4536453

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1122185, upper bound: 3.0529400
time: 0.43 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2383201, upper bound: 3.1816649
time: 0.44 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.1649871, 0.1174907, -0.2918625, 0.3971892, -0.5621763, 0.4093532
1: -0.2215959, 0.1765458, -0.4734699, 0.5811095, -0.8027054, 0.6500157
2: -0.1996021, 0.1877475, -0.3393690, 0.5553786, -0.7549807, 0.5271165
3: -0.2914082, 0.2083298, -0.6585814, 0.7684231, -1.0598313, 0.8669113
4: -0.1816220, 0.2294925, -0.4620743, 0.7093326, -0.8909546, 0.6915668

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1122185, upper bound: 3.0529400
time: 0.43 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2383201, upper bound: 3.1816649
time: 0.46 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.2107248, 0.2733173, -0.1569118, 0.1010650, -0.3117898, 0.4302291
1: -0.3105044, 0.3781556, -0.2085741, 0.1534233, -0.4639277, 0.5867297
2: -0.2597608, 0.4037241, -0.1883991, 0.1622353, -0.4219961, 0.5921233
3: -0.4464498, 0.4796840, -0.2698923, 0.1826724, -0.6291223, 0.7495763
4: -0.2856537, 0.5048870, -0.1733955, 0.1943219, -0.4799756, 0.6782825

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 17

Time for candidate selection: 4.35 seconds

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2903490, upper bound: 3.2808171
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2948815, upper bound: 3.2955185
time: 0.43 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.2559387, 0.3415756, -0.1805636, 0.1780242, -0.4339629, 0.5221391
1: -0.3946697, 0.4863652, -0.2509359, 0.2545417, -0.6492113, 0.7373011
2: -0.2981569, 0.5062003, -0.2206551, 0.2722895, -0.5704464, 0.7268554
3: -0.5855743, 0.6284050, -0.3329057, 0.3066460, -0.8922204, 0.9613106
4: -0.3760676, 0.6415557, -0.2074082, 0.3399702, -0.7160378, 0.8489639

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2756042, upper bound: 3.2858599
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 17

Time for candidate selection: 4.80 seconds

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2955465, upper bound: 3.2824327
time: 0.45 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3009438, upper bound: 3.3005710
time: 0.46 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.3367430, 0.4696040, -0.1491860, 0.0748662, -0.4116092, 0.6187899
1: -0.5362946, 0.6582009, -0.1935245, 0.1176153, -0.6539099, 0.8517253
2: -0.3761068, 0.7071391, -0.1785389, 0.1215313, -0.4976380, 0.8856781
3: -0.8453820, 0.8691105, -0.2488164, 0.1417401, -0.9871221, 1.1179268
4: -0.5167788, 0.9233156, -0.1633256, 0.1418400, -0.6586188, 1.0866412

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 17

Time for candidate selection: 4.23 seconds

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3067020, upper bound: 3.3087506
time: 0.45 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2912791, upper bound: 3.2850632
time: 0.52 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2959074, upper bound: 3.2994914
time: 0.53 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.4087062, 0.5647650, -0.1657470, 0.1321049, -0.5408109, 0.7305120
1: -0.6539204, 0.7966581, -0.2245950, 0.1951855, -0.8491057, 1.0212531
2: -0.4457314, 0.8629469, -0.2007766, 0.2073084, -0.6530398, 1.0637234
3: -1.0570911, 1.0608249, -0.2944825, 0.2328150, -1.2899061, 1.3553073
4: -0.6360155, 1.1408229, -0.1849585, 0.2545161, -0.8905316, 1.3257813

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2758223, upper bound: 3.2879790
time: 0.44 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 17

Time for candidate selection: 4.87 seconds

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3098656, upper bound: 3.3098627
time: 0.43 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2964541, upper bound: 3.2866471
time: 0.45 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3011626, upper bound: 3.3015638
time: 0.44 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.2559387, 0.3415756, -0.2424042, 0.3243341, -0.5802728, 0.5839797
1: -0.3946697, 0.4863652, -0.3773183, 0.4599679, -0.8546375, 0.8636836
2: -0.2981569, 0.5062003, -0.2913593, 0.4575135, -0.7556703, 0.7975596
3: -0.5855743, 0.6284050, -0.5175009, 0.5963498, -1.1819241, 1.1459059
4: -0.3760676, 0.6415557, -0.3589755, 0.5735825, -0.9496501, 1.0005312

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 17

Time for candidate selection: 4.32 seconds

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2920285, upper bound: 3.2812140
time: 0.45 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3006618, upper bound: 3.3006618
time: 0.42 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.2559387, 0.3415756, -0.3695980, 0.5150771, -0.7710158, 0.7111736
1: -0.3946697, 0.4863652, -0.6096500, 0.7350104, -1.1296802, 1.0960152
2: -0.2981569, 0.5062003, -0.4243644, 0.7154427, -1.0135995, 0.9305648
3: -0.5855743, 0.6284050, -0.8602619, 0.9872086, -1.5727830, 1.4886669
4: -0.3760676, 0.6415557, -0.5941179, 0.9258108, -1.3018783, 1.2356737

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 17

Time for candidate selection: 4.31 seconds

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2920285, upper bound: 3.2824327
time: 0.44 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3006618, upper bound: 3.3008632
time: 0.42 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.4087062, 0.5647650, -0.2424042, 0.3243341, -0.7330402, 0.8071691
1: -0.6539204, 0.7966581, -0.3773183, 0.4599679, -1.1138883, 1.1739764
2: -0.4457314, 0.8629469, -0.2913593, 0.4575135, -0.9032449, 1.1543062
3: -1.0570911, 1.0608249, -0.5175009, 0.5963498, -1.6534407, 1.5783257
4: -0.6360155, 1.1408229, -0.3589755, 0.5735825, -1.2095981, 1.4997983

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 17

Time for candidate selection: 4.30 seconds

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2919685, upper bound: 3.2854284
time: 0.43 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3006618, upper bound: 3.3014515
time: 0.48 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.4087062, 0.5647650, -0.3695980, 0.5150771, -0.9237834, 0.9343630
1: -0.6539204, 0.7966581, -0.6096500, 0.7350104, -1.3889307, 1.4063082
2: -0.4457314, 0.8629469, -0.4243644, 0.7154427, -1.1611741, 1.2873113
3: -1.0570911, 1.0608249, -0.8602619, 0.9872086, -2.0442996, 1.9210868
4: -0.6360155, 1.1408229, -0.5941179, 0.9258108, -1.5618262, 1.7349408

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 17

Time for candidate selection: 4.37 seconds

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2919685, upper bound: 3.2866222
time: 0.46 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3006618, upper bound: 3.3018006
time: 0.43 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0676768, 0.0203616, -0.2211066, 0.2775607, -0.3452376, 0.2414682
1: -0.0773866, 0.0391565, -0.3221700, 0.3887413, -0.4661279, 0.3613265
2: -0.0644345, 0.0282484, -0.2730529, 0.4024681, -0.4669026, 0.3013013
3: -0.0712000, 0.0471544, -0.4342754, 0.4909782, -0.5621781, 0.4814299
4: -0.0532835, 0.0354563, -0.2787128, 0.4985040, -0.5517876, 0.3141691

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3482816, upper bound: 3.3103211
time: 0.45 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3437078, upper bound: 3.2933470
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0676768, 0.0203616, -0.6112540, 0.8508660, -0.9185429, 0.6316155
1: -0.0773866, 0.0391565, -0.9703345, 1.2209729, -1.2983595, 1.0094907
2: -0.0644345, 0.0282484, -0.6424070, 1.3305063, -1.3949409, 0.6706554
3: -0.0712000, 0.0471544, -1.6326315, 1.6337559, -1.7049559, 1.6797860
4: -0.0532835, 0.0354563, -0.9662173, 1.7874435, -1.8407271, 1.0016735

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3482816, upper bound: 3.3103211
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3437078, upper bound: 3.2933470
time: 0.43 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.1649871, 0.1174907, -0.2211066, 0.2775607, -0.4425478, 0.3385974
1: -0.2215959, 0.1765458, -0.3221700, 0.3887413, -0.6103371, 0.4987158
2: -0.1996021, 0.1877475, -0.2730529, 0.4024681, -0.6020702, 0.4608004
3: -0.2914082, 0.2083298, -0.4342754, 0.4909782, -0.7823864, 0.6426053
4: -0.1816220, 0.2294925, -0.2787128, 0.4985040, -0.6801261, 0.5082053

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2980723, upper bound: 3.1901709
time: 0.43 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3099443, upper bound: 3.1973792
time: 0.46 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3100439, upper bound: 3.1978159
time: 0.49 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3136595, upper bound: 3.2009949
time: 0.46 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.1649871, 0.1174907, -0.6112540, 0.8508660, -1.0158532, 0.7287447
1: -0.2215959, 0.1765458, -0.9703345, 1.2209729, -1.4425688, 1.1468800
2: -0.1996021, 0.1877475, -0.6424070, 1.3305063, -1.5301085, 0.8301545
3: -0.2914082, 0.2083298, -1.6326315, 1.6337559, -1.9251641, 1.8409612
4: -0.1816220, 0.2294925, -0.9662173, 1.7874435, -1.9690655, 1.1957097

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2980723, upper bound: 3.1901709
time: 0.46 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3099443, upper bound: 3.1973792
time: 0.47 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3100439, upper bound: 3.1978159
time: 0.44 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3136595, upper bound: 3.2009949
time: 0.44 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0676768, 0.0203616, -0.4130764, 0.5761445, -0.6438213, 0.4334379
1: -0.0773866, 0.0391565, -0.6916107, 0.8024566, -0.8798432, 0.7307673
2: -0.0644345, 0.0282484, -0.4589225, 0.8798022, -0.9442367, 0.4871709
3: -0.0712000, 0.0471544, -1.1023782, 1.0841615, -1.1553615, 1.1495327
4: -0.0532835, 0.0354563, -0.6645420, 1.1869493, -1.2402328, 0.6999983

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 32

Time for candidate selection: 4.27 seconds

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2432762, upper bound: 3.2536595
time: 0.44 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2356813, upper bound: 3.2347458
time: 0.45 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0676768, 0.0203616, -2.3698783, 4.5716200, -4.6392970, 2.3902397
1: -0.0773866, 0.0391565, -3.6489408, 6.1731424, -6.2505293, 3.6880972
2: -0.0644345, 0.0282484, -2.3092916, 6.6463075, -6.7107420, 2.3375399
3: -0.0712000, 0.0471544, -6.8561640, 7.6015148, -7.6727147, 6.9033175
4: -0.0532835, 0.0354563, -3.7684946, 8.4313126, -8.4845963, 3.8039505

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 32

Time for candidate selection: 4.36 seconds

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2392990, upper bound: 3.2523148
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2312937, upper bound: 3.2332647
time: 0.46 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.1649871, 0.1174907, -0.4130764, 0.5761445, -0.7411315, 0.5305670
1: -0.2215959, 0.1765458, -0.6916107, 0.8024566, -1.0240525, 0.8681565
2: -0.1996021, 0.1877475, -0.4589225, 0.8798022, -1.0794044, 0.6466700
3: -0.2914082, 0.2083298, -1.1023782, 1.0841615, -1.3755697, 1.3107079
4: -0.1816220, 0.2294925, -0.6645420, 1.1869493, -1.3685713, 0.8940345

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=3.5238394737243652
rel_dist={0: [-3.3982329481346, 3.3982329481346003]}

## Binary search (step 2) starts
Candidate diff: 0.0625000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

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
- Time for IS candidates: 1.03 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.03
Output dim: 0, lower bound: -3.3740686, upper bound: 3.3318377
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.03
Output dim: 0, lower bound: -3.3943702, upper bound: 3.3943701

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.2895984, 0.3943615, -0.8808075, 1.5047889, -1.7943872, 1.2751689
1: -0.4578271, 0.5641531, -1.3966537, 2.1583297, -2.6161568, 1.9608067
2: -0.3370600, 0.5814233, -0.9524446, 2.2021837, -2.5392432, 1.5338678
3: -0.6758766, 0.7399411, -2.4135633, 2.7874823, -3.4633589, 3.1535044
4: -0.4528955, 0.7513722, -1.5303535, 2.8573139, -3.3102093, 2.2817256

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3169869, upper bound: 3.3169869
time: 0.43 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3169869, upper bound: 3.3169869
time: 0.44 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -1.0333478, 1.8749698, -0.9835699, 1.7383295, -2.7716773, 2.8585393
1: -1.6493901, 2.6237774, -1.5618739, 2.4618018, -4.1111917, 4.1856508
2: -1.1160958, 2.7230368, -1.0621964, 2.5254564, -3.6415520, 3.7852330
3: -2.9354391, 3.3341267, -2.7349937, 3.1533265, -6.0887647, 6.0691204
4: -1.7549448, 3.5216486, -1.6960667, 3.2828889, -5.0378327, 5.2177153

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28

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
time: 0.44 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.31 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.31
Output dim: 0, lower bound: -3.3169869, upper bound: 3.3169869
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.31
Output dim: 0, lower bound: -3.3169869, upper bound: 3.3169869
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.31
Output dim: 0, lower bound: -3.3318377, upper bound: 3.3740686
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.31
Output dim: 0, lower bound: -3.3318377, upper bound: 3.3740686

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.2895984, 0.3943615, -0.2895984, 0.3943615, -0.6839598, 0.6839598
1: -0.4578271, 0.5641531, -0.4578271, 0.5641531, -1.0219798, 1.0219798
2: -0.3370600, 0.5814233, -0.3370600, 0.5814233, -0.9184830, 0.9184831
3: -0.6758766, 0.7399411, -0.6758766, 0.7399411, -1.4158176, 1.4158175
4: -0.4528955, 0.7513722, -0.4528955, 0.7513722, -1.2042676, 1.2042676

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3169397, upper bound: 3.3169869
time: 0.38 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3169869, upper bound: 3.3169869
time: 0.51 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.2895984, 0.3943615, -1.0233476, 1.8376125, -2.1272104, 1.4177091
1: -0.4578271, 0.5641531, -1.6329428, 2.5899200, -3.0477471, 2.1970959
2: -0.3370600, 0.5814233, -1.1025296, 2.6656742, -3.0027339, 1.6839529
3: -0.6758766, 0.7399411, -2.8889070, 3.3017845, -3.9776611, 3.6288481
4: -0.4528955, 0.7513722, -1.7341112, 3.4505391, -3.9034343, 2.4854827

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3169397, upper bound: 3.3318343
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3169869, upper bound: 3.3318343
time: 0.43 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -1.0333478, 1.8749698, -0.2895984, 0.3943615, -1.4277093, 2.1645677
1: -1.6493901, 2.6237774, -0.4578271, 0.5641531, -2.2135432, 3.0816045
2: -1.1160958, 2.7230368, -0.3370600, 0.5814233, -1.6975191, 3.0600965
3: -2.9354391, 3.3341267, -0.6758766, 0.7399411, -3.6753798, 4.0100031
4: -1.7549448, 3.5216486, -0.4528955, 0.7513722, -2.5063167, 3.9745436

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3171830, upper bound: 3.3591355
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3044364, upper bound: 3.3113877
time: 0.46 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -1.0333478, 1.8749698, -1.0299754, 1.8632205, -2.8965681, 2.9049451
1: -1.6493901, 2.6237774, -1.6423873, 2.6173930, -4.2667828, 4.2661643
2: -1.1160958, 2.7230368, -1.1113069, 2.6993732, -3.8154690, 3.8343437
3: -2.9354391, 3.3341267, -2.9127293, 3.3258715, -6.2613106, 6.2468557
4: -1.7549448, 3.5216486, -1.7470773, 3.4881601, -5.2431049, 5.2687259

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3171830, upper bound: 3.3617727
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3044364, upper bound: 3.3205784
time: 0.43 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.22 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.22
Output dim: 0, lower bound: -3.3169397, upper bound: 3.3169869
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.22
Output dim: 0, lower bound: -3.3169869, upper bound: 3.3169869
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.22
Output dim: 0, lower bound: -3.3169397, upper bound: 3.3318343
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.22
Output dim: 0, lower bound: -3.3169869, upper bound: 3.3318343
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.22
Output dim: 0, lower bound: -3.3171830, upper bound: 3.3591355
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.22
Output dim: 0, lower bound: -3.3044364, upper bound: 3.3113877
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.22
Output dim: 0, lower bound: -3.3171830, upper bound: 3.3617727
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.22
Output dim: 0, lower bound: -3.3044364, upper bound: 3.3205784

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1912085, 0.2109509, -0.2089019, 0.2612902, -0.4524986, 0.4198529
1: -0.2706243, 0.2982620, -0.3096748, 0.3649721, -0.6355963, 0.6079369
2: -0.2345207, 0.3197280, -0.2579902, 0.3913511, -0.6258718, 0.5777181
3: -0.3691734, 0.3672814, -0.4369481, 0.4640577, -0.8332311, 0.8042295
4: -0.2326370, 0.4024982, -0.2882957, 0.4949898, -0.7276268, 0.6907939

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2713582, upper bound: 3.2643254
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2323659, upper bound: 3.1799445
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1983875, upper bound: 3.1701139
time: 0.35 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.4519231, 0.6203508, -0.2139930, 0.2726936, -0.7246166, 0.8343437
1: -0.7154577, 0.8934209, -0.3252755, 0.3844926, -1.0999503, 1.2186964
2: -0.4850966, 0.9528847, -0.2639961, 0.4031157, -0.8882121, 1.2168808
3: -1.1693269, 1.1907833, -0.4551041, 0.4892960, -1.6586229, 1.6458874
4: -0.7152326, 1.2680461, -0.3003655, 0.5078025, -1.2230351, 1.5684116

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3054507, upper bound: 3.3087832
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3098655, upper bound: 3.3098656
time: 0.40 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.1912085, 0.2109509, -0.8387710, 1.4471810, -1.6383895, 1.0497218
1: -0.2706243, 0.2982620, -1.3424455, 2.0324211, -2.3030453, 1.6407075
2: -0.2345207, 0.3197280, -0.8940362, 2.1313763, -2.3658969, 1.2137641
3: -0.3691734, 0.3672814, -2.3696289, 2.6029589, -2.9721324, 2.7369103
4: -0.2326370, 0.4024982, -1.3887299, 2.7785869, -3.0112238, 1.7912281

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3207095, upper bound: 3.3109368
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3686039, upper bound: 3.3249191
time: 0.43 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.4519231, 0.6203508, -0.7688240, 1.3056786, -1.7576016, 1.3891747
1: -0.7154577, 0.8934209, -1.2446048, 1.8586768, -2.5741343, 2.1380258
2: -0.4850966, 0.9528847, -0.8271433, 1.9219395, -2.4070358, 1.7800277
3: -1.1693269, 1.1907833, -2.1716218, 2.3883061, -3.5576329, 3.3624051
4: -0.7152326, 1.2680461, -1.2829863, 2.5161824, -3.2314150, 2.5510321

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3591355, upper bound: 3.3171830
time: 0.44 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3113877, upper bound: 3.3044364
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.3753012, 0.5288189, -0.2162850, 0.2847219, -0.6600231, 0.7451039
1: -0.6084703, 0.7558654, -0.3347894, 0.4040710, -1.0125411, 1.0906547
2: -0.4198778, 0.7994800, -0.2651390, 0.4216872, -0.8415650, 1.0646191
3: -0.9523419, 0.9947283, -0.4710898, 0.5189832, -1.4713252, 1.4658180
4: -0.5820951, 1.0479591, -0.3187409, 0.5333543, -1.1154494, 1.3666999

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2757767, upper bound: 3.3252739
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3086417, upper bound: 3.3498469
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3044364, upper bound: 3.3113877
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3044364, upper bound: 3.3113877
time: 0.46 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.6188753, 0.9653776, -0.2077290, 0.2577925, -0.8766679, 1.1731066
1: -1.0178568, 1.3381594, -0.3075950, 0.3612317, -1.3790884, 1.6457543
2: -0.6695868, 1.4706283, -0.2561093, 0.3884775, -1.0580643, 1.7267376
3: -1.7465825, 1.7551888, -0.4374343, 0.4596128, -2.2061954, 2.1926231
4: -1.0122970, 1.9336305, -0.2893268, 0.4917006, -1.5039977, 2.2229574

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3044364, upper bound: 3.3113877
time: 0.45 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3044364, upper bound: 3.3113877
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.3753012, 0.5288189, -0.8684130, 1.5168421, -1.8921428, 1.3972318
1: -0.6084703, 0.7558654, -1.3934584, 2.1353853, -2.7438555, 2.1493237
2: -0.4198778, 0.7994800, -0.9286463, 2.2242820, -2.6441598, 1.7281262
3: -0.9523419, 0.9947283, -2.4577522, 2.7305937, -3.6829350, 3.4524796
4: -0.5820951, 1.0479591, -1.4501021, 2.8982253, -3.4803205, 2.4980609

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3205783, upper bound: 3.3205783
time: 0.45 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3205783, upper bound: 3.3205783
time: 0.44 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.6188753, 0.9653776, -0.7438566, 1.2501802, -1.8690555, 1.7092342
1: -1.0178568, 1.3381594, -1.2127101, 1.7404547, -2.7583115, 2.5508692
2: -0.6695868, 1.4706283, -0.8007102, 1.8587987, -2.5283856, 2.2713385
3: -1.7465825, 1.7551888, -2.1174743, 2.2482402, -3.9948227, 3.8726623
4: -1.0122970, 1.9336305, -1.2348790, 2.4262033, -3.4385004, 3.1685090

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3205783, upper bound: 3.3205783
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3205783, upper bound: 3.3205783
time: 0.45 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.04 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.04
Output dim: 0, lower bound: -3.2323659, upper bound: 3.1799445
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.04
Output dim: 0, lower bound: -3.1983875, upper bound: 3.1701139
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.04
Output dim: 0, lower bound: -3.3054507, upper bound: 3.3087832
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.04
Output dim: 0, lower bound: -3.3098655, upper bound: 3.3098656
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.04
Output dim: 0, lower bound: -3.3207095, upper bound: 3.3109368
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.04
Output dim: 0, lower bound: -3.3686039, upper bound: 3.3249191
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.04
Output dim: 0, lower bound: -3.3591355, upper bound: 3.3171830
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.04
Output dim: 0, lower bound: -3.3113877, upper bound: 3.3044364
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.04
Output dim: 0, lower bound: -3.3044364, upper bound: 3.3113877
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.04
Output dim: 0, lower bound: -3.3044364, upper bound: 3.3113877
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.04
Output dim: 0, lower bound: -3.3044364, upper bound: 3.3113877
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.04
Output dim: 0, lower bound: -3.3044364, upper bound: 3.3113877
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.04
Output dim: 0, lower bound: -3.3205783, upper bound: 3.3205783
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.04
Output dim: 0, lower bound: -3.3205783, upper bound: 3.3205783
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.04
Output dim: 0, lower bound: -3.3205783, upper bound: 3.3205783
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.04
Output dim: 0, lower bound: -3.3205783, upper bound: 3.3205783

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1703591, 0.1513717, -0.1394853, 0.0392893, -0.2096484, 0.2908571
1: -0.2340399, 0.2190559, -0.1906739, 0.0674425, -0.3014824, 0.4097298
2: -0.2074144, 0.2342778, -0.1791689, 0.0553716, -0.2627860, 0.4134468
3: -0.3079444, 0.2615820, -0.1800014, 0.0894042, -0.3973486, 0.4415834
4: -0.1916540, 0.2903985, -0.1092302, 0.0735825, -0.2652365, 0.3996287

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1983875, upper bound: 3.1701139
time: 0.46 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1983875, upper bound: 3.1701139
time: 0.35 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1695808, 0.1379753, -0.1747664, 0.1573851, -0.3269659, 0.3127417
1: -0.2305985, 0.2023515, -0.2405900, 0.2262007, -0.4567993, 0.4429415
2: -0.2059347, 0.2169279, -0.2129948, 0.2443947, -0.4503294, 0.4299227
3: -0.3048239, 0.2396203, -0.3196438, 0.2689870, -0.5738109, 0.5592641
4: -0.1882927, 0.2685840, -0.1956582, 0.3052854, -0.4935781, 0.4642422

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9325236, upper bound: 2.9788020
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8846540, upper bound: 2.8685681
time: 0.37 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.4200564, 0.5764957, -0.1694381, 0.1335216, -0.5535780, 0.7459338
1: -0.6649380, 0.8279919, -0.2286797, 0.1980378, -0.8629758, 1.0566716
2: -0.4546110, 0.8816518, -0.2057150, 0.2046002, -0.6592112, 1.0873668
3: -1.0772572, 1.1003878, -0.3019956, 0.2358920, -1.3131492, 1.4023834
4: -0.6603726, 1.1683871, -0.1904275, 0.2487192, -0.9090918, 1.3588146

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3043691, upper bound: 3.3043691
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3043691, upper bound: 3.3087832
time: 0.44 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.3453564, 0.4772129, -0.1878572, 0.1977879, -0.5431442, 0.6650701
1: -0.5476037, 0.6854626, -0.2627091, 0.2706024, -0.8182061, 0.9481717
2: -0.3822781, 0.7141160, -0.2296301, 0.2924444, -0.6747225, 0.9437460
3: -0.8585606, 0.9054945, -0.3440384, 0.3256769, -1.1842375, 1.2495329
4: -0.5394703, 0.9371172, -0.2155138, 0.3579382, -0.8974085, 1.1526310

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3087832, upper bound: 3.3054507
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3087832, upper bound: 3.3098656
time: 0.43 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.1846969, 0.1896814, -0.4892117, 0.7003186, -0.8850155, 0.6788931
1: -0.2575498, 0.2694632, -0.7909114, 0.9510068, -1.2085565, 1.0603747
2: -0.2257361, 0.2890286, -0.5302792, 1.0910958, -1.3168318, 0.8193078
3: -0.3443667, 0.3268602, -1.3417283, 1.2634159, -1.6077826, 1.6685884
4: -0.2160943, 0.3629016, -0.7672535, 1.4652178, -1.6813121, 1.1301551

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3207095, upper bound: 3.3086230
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3207095, upper bound: 3.3109368
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.1631718, 0.1269484, -0.5268661, 0.7673208, -0.9304926, 0.6538145
1: -0.2201545, 0.1873775, -0.8521317, 1.0699952, -1.2901497, 1.0395089
2: -0.1971339, 0.1995453, -0.5639910, 1.1860915, -1.3832254, 0.7635362
3: -0.2890078, 0.2231394, -1.4545635, 1.4130675, -1.7020752, 1.6777029
4: -0.1811372, 0.2442653, -0.8294135, 1.5811107, -1.7622479, 1.0736789

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3433368, upper bound: 3.2984795
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3677805, upper bound: 3.3203509
time: 0.45 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3677806, upper bound: 3.3249192
time: 0.44 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.3771043, 0.5194778, -0.3133364, 0.4384587, -0.8155630, 0.8328142
1: -0.5969394, 0.7479213, -0.5069219, 0.6292082, -1.2261477, 1.2548432
2: -0.4127874, 0.7854912, -0.3592748, 0.6541829, -1.0669703, 1.1447661
3: -0.9499584, 0.9901552, -0.7626505, 0.8189385, -1.7688968, 1.7528057
4: -0.5909911, 1.0352731, -0.4751250, 0.8419933, -1.4329840, 1.5103980

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3498469, upper bound: 3.3086417
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3113878, upper bound: 3.3044364
time: 0.43 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3113878, upper bound: 3.3044364
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.3400626, 0.4655252, -0.4307776, 0.5922384, -0.9323010, 0.8963028
1: -0.5467659, 0.6746459, -0.7231025, 0.8382097, -1.3849757, 1.3977484
2: -0.3808751, 0.6913177, -0.4832181, 0.8973091, -1.2781843, 1.1745358
3: -0.8449534, 0.8927982, -1.1504639, 1.1237490, -1.9687024, 2.0432620
4: -0.5362781, 0.9068828, -0.6869715, 1.2096882, -1.7459662, 1.5938540

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3041970, upper bound: 3.2855741
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3113877, upper bound: 3.3044364
time: 0.44 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3113878, upper bound: 3.3044364
time: 0.45 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.3753012, 0.5288189, -0.1562187, 0.0892744, -0.4645756, 0.6850376
1: -0.6084703, 0.7558654, -0.2067900, 0.1396068, -0.7480771, 0.9626554
2: -0.4198778, 0.7994800, -0.1889970, 0.1377370, -0.5576147, 0.9884770
3: -0.9523419, 0.9947283, -0.2654234, 0.1652304, -1.1175723, 1.2601517
4: -0.5820951, 1.0479591, -0.1719097, 0.1611379, -0.7432331, 1.2198688

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2340108, upper bound: 3.2894622
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3171830, upper bound: 3.3591355
time: 0.46 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.3753012, 0.5288189, -0.1972768, 0.2288733, -0.6041744, 0.7260957
1: -0.6084703, 0.7558654, -0.2833577, 0.3218512, -0.9303215, 1.0392231
2: -0.4198778, 0.7994800, -0.2425594, 0.3484467, -0.7683244, 1.0420394
3: -0.9523419, 0.9947283, -0.3993823, 0.4013456, -1.3536874, 1.3941106
4: -0.5820951, 1.0479591, -0.2542248, 0.4395830, -1.0216781, 1.3021839

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2340108, upper bound: 3.2894622
time: 0.44 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3171830, upper bound: 3.3591355
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.6188753, 0.9653776, -0.1434819, 0.0705145, -0.6893898, 1.1088594
1: -1.0178568, 1.3381594, -0.2003747, 0.1103751, -1.1282319, 1.5385342
2: -0.6695868, 1.4706283, -0.1849771, 0.1064460, -0.7760327, 1.6556053
3: -1.7465825, 1.7551888, -0.2004221, 0.1352789, -1.8818613, 1.9556110
4: -1.0122970, 1.9336305, -0.1163127, 0.1330654, -1.1453625, 2.0499432

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2913383, upper bound: 3.3113877
time: 0.45 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3044364, upper bound: 3.3113440
time: 0.45 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.6188753, 0.9653776, -0.1972768, 0.2288733, -0.8477486, 1.1626544
1: -1.0178568, 1.3381594, -0.2833577, 0.3218512, -1.3397081, 1.6215172
2: -0.6695868, 1.4706283, -0.2425594, 0.3484467, -1.0180335, 1.7131877
3: -1.7465825, 1.7551888, -0.3993823, 0.4013456, -2.1479280, 2.1545711
4: -1.0122970, 1.9336305, -0.2542248, 0.4395830, -1.4518801, 2.1878552

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2913383, upper bound: 3.3113877
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3044364, upper bound: 3.3113440
time: 0.44 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.3753012, 0.5288189, -0.3753012, 0.5288189, -0.9041201, 0.9041200
1: -0.6084703, 0.7558654, -0.6084703, 0.7558654, -1.3643357, 1.3643357
2: -0.4198778, 0.7994800, -0.4198778, 0.7994800, -1.2193577, 1.2193577
3: -0.9523419, 0.9947283, -0.9523419, 0.9947283, -1.9470696, 1.9470699
4: -0.5820951, 1.0479591, -0.5820951, 1.0479591, -1.6300540, 1.6300540

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3277742, upper bound: 3.3617726
time: 0.45 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3223838, upper bound: 3.3347859
time: 0.47 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.3753012, 0.5288189, -0.6188753, 0.9653776, -1.3406787, 1.1476941
1: -0.6084703, 0.7558654, -1.0178568, 1.3381594, -1.9466296, 1.7737222
2: -0.4198778, 0.7994800, -0.6695868, 1.4706283, -1.8905060, 1.4690667
3: -0.9523419, 0.9947283, -1.7465825, 1.7551888, -2.7075305, 2.7413108
4: -0.5820951, 1.0479591, -1.0122970, 1.9336305, -2.5157256, 2.0602558

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3277742, upper bound: 3.3617726
time: 0.45 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3223838, upper bound: 3.3347859
time: 0.47 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.6188753, 0.9653776, -0.3741371, 0.5269459, -1.1458211, 1.3395146
1: -1.0178568, 1.3381594, -0.6063213, 0.7530777, -1.7709343, 1.9444808
2: -0.6695868, 1.4706283, -0.4185858, 0.7965392, -1.4661260, 1.8892140
3: -1.7465825, 1.7551888, -0.9483701, 0.9909983, -2.7375808, 2.7035589
4: -1.0122970, 1.9336305, -0.5800145, 1.0438086, -2.0561056, 2.5136445

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3134740, upper bound: 3.3133501
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3135192, upper bound: 3.3135192
time: 0.45 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.6188753, 0.9653776, -0.6188753, 0.9653776, -1.5842528, 1.5842528
1: -1.0178568, 1.3381594, -1.0178568, 1.3381594, -2.3560159, 2.3560159
2: -0.6695868, 1.4706283, -0.6695868, 1.4706283, -2.1402152, 2.1402152
3: -1.7465825, 1.7551888, -1.7465825, 1.7551888, -3.5017715, 3.5017715
4: -1.0122970, 1.9336305, -1.0122970, 1.9336305, -2.9459276, 2.9459276

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3134740, upper bound: 3.3133501
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3135192, upper bound: 3.3135192
time: 0.47 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.18 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -3.1983875, upper bound: 3.1701139
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -3.1983875, upper bound: 3.1701139
IS_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.18
Output dim: 0, lower bound: -2.9325236, upper bound: 2.9788020
IS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.18
Output dim: 0, lower bound: -2.8846540, upper bound: 2.8685681
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -3.3043691, upper bound: 3.3043691
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -3.3043691, upper bound: 3.3087832
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -3.3087832, upper bound: 3.3054507
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -3.3087832, upper bound: 3.3098656
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -3.3207095, upper bound: 3.3086230
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -3.3207095, upper bound: 3.3109368
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -3.3677805, upper bound: 3.3203509
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -3.3677806, upper bound: 3.3249192
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -3.3113878, upper bound: 3.3044364
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -3.3113878, upper bound: 3.3044364
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -3.3113877, upper bound: 3.3044364
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -3.3113878, upper bound: 3.3044364
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -3.2340108, upper bound: 3.2894622
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -3.3171830, upper bound: 3.3591355
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -3.2340108, upper bound: 3.2894622
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -3.3171830, upper bound: 3.3591355
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -3.2913383, upper bound: 3.3113877
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -3.3044364, upper bound: 3.3113440
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -3.2913383, upper bound: 3.3113877
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -3.3044364, upper bound: 3.3113440
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -3.3277742, upper bound: 3.3617726
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -3.3223838, upper bound: 3.3347859
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -3.3277742, upper bound: 3.3617726
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -3.3223838, upper bound: 3.3347859
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -3.3134740, upper bound: 3.3133501
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -3.3135192, upper bound: 3.3135192
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -3.3134740, upper bound: 3.3133501
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -3.3135192, upper bound: 3.3135192

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0676768, 0.0203616, -0.1394853, 0.0392893, -0.1069662, 0.1598469
1: -0.0773866, 0.0391565, -0.1906739, 0.0674425, -0.1448292, 0.2298305
2: -0.0644345, 0.0282484, -0.1791689, 0.0553716, -0.1198061, 0.2074173
3: -0.0712000, 0.0471544, -0.1800014, 0.0894042, -0.1606041, 0.2271558
4: -0.0532835, 0.0354563, -0.1092302, 0.0735825, -0.1268660, 0.1446865

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2307899, upper bound: 3.1781943
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2307025, upper bound: 3.1781355
time: 0.41 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.1649871, 0.1174907, -0.1394853, 0.0392893, -0.2042764, 0.2569761
1: -0.2215959, 0.1765458, -0.1906739, 0.0674425, -0.2890384, 0.3672197
2: -0.1996021, 0.1877475, -0.1791689, 0.0553716, -0.2549737, 0.3669164
3: -0.2914082, 0.2083298, -0.1800014, 0.0894042, -0.3808124, 0.3883313
4: -0.1816220, 0.2294925, -0.1092302, 0.0735825, -0.2552046, 0.3387226

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2307899, upper bound: 3.1781943
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2307025, upper bound: 3.1781355
time: 0.42 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.2559387, 0.3415756, -0.1694381, 0.1335216, -0.3894603, 0.5110136
1: -0.3946697, 0.4863652, -0.2286797, 0.1980378, -0.5927075, 0.7150449
2: -0.2981569, 0.5062003, -0.2057150, 0.2046002, -0.5027570, 0.7119154
3: -0.5855743, 0.6284050, -0.3019956, 0.2358920, -0.8214663, 0.9304006
4: -0.3760676, 0.6415557, -0.1904275, 0.2487192, -0.6247868, 0.8319832

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3041493, upper bound: 3.3042061
time: 0.43 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3041493, upper bound: 3.3043367
time: 0.45 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.4087062, 0.5647650, -0.1694381, 0.1335216, -0.5422277, 0.7342031
1: -0.6539204, 0.7966581, -0.2286797, 0.1980378, -0.8519583, 1.0253378
2: -0.4457314, 0.8629469, -0.2057150, 0.2046002, -0.6503316, 1.0686619
3: -1.0570911, 1.0608249, -0.3019956, 0.2358920, -1.2929831, 1.3628205
4: -0.6360155, 1.1408229, -0.1904275, 0.2487192, -0.8847347, 1.3312504

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3041493, upper bound: 3.3086207
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3041493, upper bound: 3.3086207
time: 0.44 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.2559387, 0.3415756, -0.1878572, 0.1977879, -0.4537265, 0.5294328
1: -0.3946697, 0.4863652, -0.2627091, 0.2706024, -0.6652721, 0.7490743
2: -0.2981569, 0.5062003, -0.2296301, 0.2924444, -0.5906012, 0.7358304
3: -0.5855743, 0.6284050, -0.3440384, 0.3256769, -0.9112513, 0.9724433
4: -0.3760676, 0.6415557, -0.2155138, 0.3579382, -0.7340057, 0.8570695

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3041493, upper bound: 3.3054478
time: 0.44 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3041493, upper bound: 3.3054507
time: 0.41 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.4069480, 0.5625565, -0.1878572, 0.1977879, -0.6047359, 0.7504137
1: -0.6515671, 0.7928960, -0.2627091, 0.2706024, -0.9221694, 1.0556052
2: -0.4441905, 0.8605536, -0.2296301, 0.2924444, -0.7366349, 1.0901837
3: -1.0537879, 1.0555198, -0.3440384, 0.3256769, -1.3794649, 1.3995582
4: -0.6327837, 1.1381221, -0.2155138, 0.3579382, -0.9907218, 1.3536359

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3041493, upper bound: 3.3098627
time: 0.44 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3041493, upper bound: 3.3098627
time: 0.42 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1506878, 0.0907005, -0.4892117, 0.7003186, -0.8510064, 0.5799121
1: -0.2101478, 0.1401098, -0.7909114, 0.9510068, -1.1611546, 0.9310212
2: -0.1937296, 0.1382380, -0.5302792, 1.0910958, -1.2848253, 0.6685172
3: -0.2145611, 0.1737585, -1.3417283, 1.2634159, -1.4779770, 1.5154867
4: -0.1262522, 0.1747571, -0.7672535, 1.4652178, -1.5914700, 0.9420105

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3194880, upper bound: 3.3074085
time: 0.43 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3194880, upper bound: 3.3086230
time: 0.44 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.1706024, 0.1344135, -0.4892117, 0.7003186, -0.8709211, 0.6236252
1: -0.2284564, 0.1925970, -0.7909114, 0.9510068, -1.1794631, 0.9835083
2: -0.2065775, 0.1987212, -0.5302792, 1.0910958, -1.2976732, 0.7290004
3: -0.2968557, 0.2297280, -1.3417283, 1.2634159, -1.5602716, 1.5714563
4: -0.1908983, 0.2396057, -0.7672535, 1.4652178, -1.6561161, 1.0068589

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3194880, upper bound: 3.3096523
time: 0.43 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3194880, upper bound: 3.3109368
time: 0.42 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.1506860, 0.0906823, -0.5268661, 0.7673208, -0.9180068, 0.6175484
1: -0.2101426, 0.1400905, -0.8521317, 1.0699952, -1.2801378, 0.9922221
2: -0.1937267, 0.1382093, -0.5639910, 1.1860915, -1.3798182, 0.7022001
3: -0.2145486, 0.1737360, -1.4545635, 1.4130675, -1.6276160, 1.6282994
4: -0.1262489, 0.1747204, -0.8294135, 1.5811107, -1.7073596, 1.0041338

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3194880, upper bound: 3.3203509
time: 0.47 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3194880, upper bound: 3.3203508
time: 0.45 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.1706024, 0.1344135, -0.5268661, 0.7673208, -0.9379233, 0.6612795
1: -0.2284564, 0.1925970, -0.8521317, 1.0699952, -1.2984515, 1.0447285
2: -0.2065775, 0.1987212, -0.5639910, 1.1860915, -1.3926690, 0.7627121
3: -0.2968557, 0.2297280, -1.4545635, 1.4130675, -1.7099231, 1.6842915
4: -0.1908983, 0.2396057, -0.8294135, 1.5811107, -1.7720090, 1.0690192

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3194880, upper bound: 3.3134235
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3194880, upper bound: 3.3143501
time: 0.44 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.1850611, 0.2096545, -0.3133364, 0.4384587, -0.6235198, 0.5229909
1: -0.2660490, 0.2913542, -0.5069219, 0.6292082, -0.8952572, 0.7982761
2: -0.2290718, 0.3008571, -0.3592748, 0.6541829, -0.8832547, 0.6601319
3: -0.3428191, 0.3601812, -0.7626505, 0.8189385, -1.1617576, 1.1228317
4: -0.2267372, 0.3690930, -0.4751250, 0.8419933, -1.0687305, 0.8442179

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2894622, upper bound: 3.2340108
time: 0.43 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2894622, upper bound: 3.3171830
time: 0.46 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.3088257, 0.4204044, -0.3133364, 0.4384587, -0.7472844, 0.7337407
1: -0.4969721, 0.6138726, -0.5069219, 0.6292082, -1.1261803, 1.1207945
2: -0.3497591, 0.6176917, -0.3592748, 0.6541829, -1.0039420, 0.9769665
3: -0.7490399, 0.8086340, -0.7626505, 0.8189385, -1.5679784, 1.5712845
4: -0.4839897, 0.8007323, -0.4751250, 0.8419933, -1.3259827, 1.2758572

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2894622, upper bound: 3.2340108
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2894622, upper bound: 3.3171830
time: 0.45 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.1772927, 0.1800812, -0.4307776, 0.5922384, -0.7695311, 0.6108587
1: -0.2496039, 0.2482904, -0.7231025, 0.8382097, -1.0878136, 0.9713929
2: -0.2176473, 0.2596843, -0.4832181, 0.8973091, -1.1149565, 0.7429023
3: -0.3206638, 0.3015586, -1.1504639, 1.1237490, -1.4444128, 1.4520223
4: -0.2078230, 0.3169680, -0.6869715, 1.2096882, -1.4175112, 1.0039392

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3113878, upper bound: 3.2913383
time: 0.43 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3113440, upper bound: 3.3044364
time: 0.44 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.3088257, 0.4204044, -0.4307776, 0.5922384, -0.9010641, 0.8511819
1: -0.4969721, 0.6138726, -0.7231025, 0.8382097, -1.3351818, 1.3369751
2: -0.3497591, 0.6176917, -0.4832181, 0.8973091, -1.2470682, 1.1009097
3: -0.7490399, 0.8086340, -1.1504639, 1.1237490, -1.8727888, 1.9590975
4: -0.4839897, 0.8007323, -0.6869715, 1.2096882, -1.6936778, 1.4877036

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3113878, upper bound: 3.2913383
time: 0.43 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3113440, upper bound: 3.3044364
time: 0.46 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.2211066, 0.2775607, -0.1394853, 0.0392893, -0.2603960, 0.4170460
1: -0.3221700, 0.3887413, -0.1906739, 0.0674425, -0.3896126, 0.5794152
2: -0.2730529, 0.4024681, -0.1791689, 0.0553716, -0.3284245, 0.5816370
3: -0.4342754, 0.4909782, -0.1800014, 0.0894042, -0.5236796, 0.6709796
4: -0.2787128, 0.4985040, -0.1092302, 0.0735825, -0.3522954, 0.6077343

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3205816, upper bound: 3.3601946
time: 0.50 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3206436, upper bound: 3.3683324
time: 0.44 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3250562, upper bound: 3.3691405
time: 0.46 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.6227801, 0.8648329, -0.1419401, 0.0566032, -0.6793833, 1.0067730
1: -0.9949624, 1.2503651, -0.1960979, 0.0908820, -1.0858443, 1.4464630
2: -0.6539169, 1.3625121, -0.1827588, 0.0805571, -0.7344741, 1.5452709
3: -1.6873388, 1.6740556, -0.1903948, 0.1152245, -1.8025632, 1.8644505
4: -0.9924154, 1.8411534, -0.1134843, 0.1015033, -1.0939188, 1.9546376

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3052482, upper bound: 3.3470488
time: 0.44 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3204385, upper bound: 3.3667918
time: 0.45 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3247598, upper bound: 3.3675337
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.2211066, 0.2775607, -0.1747664, 0.1573851, -0.3784918, 0.4523271
1: -0.3221700, 0.3887413, -0.2405900, 0.2262007, -0.5483707, 0.6293312
2: -0.2730529, 0.4024681, -0.2129948, 0.2443947, -0.5174476, 0.6154629
3: -0.4342754, 0.4909782, -0.3196438, 0.2689870, -0.7032624, 0.8106220
4: -0.2787128, 0.4985040, -0.1956582, 0.3052854, -0.5839983, 0.6941622

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2288437, upper bound: 3.2837066
time: 0.44 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 21
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 6

Time for candidate selection: 5.32 seconds

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 9

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 21

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2125103, upper bound: 3.2664578
time: 0.45 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2098741, upper bound: 3.2644273
time: 0.47 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.6227801, 0.8648329, -0.1783098, 0.1693269, -0.7921069, 1.0431427
1: -0.9949624, 1.2503651, -0.2487985, 0.2390516, -1.2340138, 1.4991636
2: -0.6539169, 1.3625121, -0.2178040, 0.2593426, -0.9132593, 1.5803161
3: -1.6873388, 1.6740556, -0.3296666, 0.2853321, -1.9726709, 2.0037222
4: -0.9924154, 1.8411534, -0.2006868, 0.3225284, -1.3149438, 2.0418401

Time for backsubstitution: 2.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2757767, upper bound: 3.3252739
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2635130, upper bound: 3.3100920
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3170610, upper bound: 3.3590794
time: 0.44 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.7091317, 1.1715200, -0.1385692, 0.0370639, -0.7461956, 1.3100891
1: -1.1533722, 1.6230139, -0.1894392, 0.0647501, -1.2181224, 1.8124530
2: -0.7592056, 1.7569205, -0.1777008, 0.0550650, -0.8142706, 1.9346212
3: -2.0259933, 2.1003478, -0.1808517, 0.0844789, -2.1104722, 2.2811995
4: -1.1636406, 2.2863564, -0.1074460, 0.0761417, -1.2397822, 2.3938024

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2850248, upper bound: 3.3043757
time: 0.46 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 21
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 32

Time for candidate selection: 5.15 seconds

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 9

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 21

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2757392, upper bound: 3.2994781
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2805548, upper bound: 3.3005894
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.4738564, 0.6639866, -0.1434819, 0.0705145, -0.5443709, 0.8074685
1: -0.7879710, 0.9146689, -0.2003747, 0.1103751, -0.8983460, 1.1150435
2: -0.5224863, 1.0312799, -0.1849771, 0.1064460, -0.6289323, 1.2162570
3: -1.3006039, 1.2264051, -0.2004221, 0.1352789, -1.4358828, 1.4268273
4: -0.7528779, 1.3891870, -0.1163127, 0.1330654, -0.8859433, 1.5054997

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2867448, upper bound: 3.3037141
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 21
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 32

Time for candidate selection: 5.28 seconds

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 9

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2806294, upper bound: 3.2968412
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 21

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2760943, upper bound: 3.2994492
time: 0.44 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3038872, upper bound: 3.3007717
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.7091317, 1.1715200, -0.1822311, 0.1884407, -0.8975725, 1.3537512
1: -1.1533722, 1.6230139, -0.2565054, 0.2668566, -1.4202287, 1.8795192
2: -0.7592056, 1.7569205, -0.2230698, 0.2905326, -1.0497382, 1.9799902
3: -2.0259933, 2.1003478, -0.3440322, 0.3242811, -2.3502743, 2.4443798
4: -1.1636406, 2.2863564, -0.2159932, 0.3641888, -1.5278294, 2.5023496

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2840318, upper bound: 3.3041970
time: 0.45 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 21
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 6

Time for candidate selection: 5.18 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 21

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2727248, upper bound: 3.2991534
time: 0.47 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2805548, upper bound: 3.3005894
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.4738564, 0.6639866, -0.1972768, 0.2288733, -0.7027296, 0.8612634
1: -0.7879710, 0.9146689, -0.2833577, 0.3218512, -1.1098222, 1.1980265
2: -0.5224863, 1.0312799, -0.2425594, 0.3484467, -0.8709329, 1.2738392
3: -1.3006039, 1.2264051, -0.3993823, 0.4013456, -1.7019494, 1.6257875
4: -0.7528779, 1.3891870, -0.2542248, 0.4395830, -1.1924607, 1.6434118

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2855741, upper bound: 3.3034933
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 21
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 6

Time for candidate selection: 5.27 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 21

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2726827, upper bound: 3.2990963
time: 0.45 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2984112, upper bound: 3.3007717
time: 0.50 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.3456418, 0.4810718, -0.2707007, 0.3777031, -0.7233449, 0.7517725
1: -0.5610024, 0.6868975, -0.4357001, 0.5446322, -1.1056343, 1.1225975
2: -0.3901013, 0.7285742, -0.3177954, 0.5568109, -0.9469122, 1.0463696
3: -0.8672440, 0.8995361, -0.6296080, 0.7041204, -1.5713644, 1.5291440
4: -0.5257008, 0.9459569, -0.4058248, 0.7071536, -1.2328544, 1.3517816

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3544338, upper bound: 3.3824468
time: 0.43 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3544338, upper bound: 3.3824556
time: 0.49 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.2695478, 0.3760729, -0.3753012, 0.5288189, -0.7983667, 0.7513741
1: -0.4326958, 0.5375808, -0.6084703, 0.7558654, -1.1885612, 1.1460508
2: -0.3200988, 0.5548421, -0.4198778, 0.7994800, -1.1195787, 0.9747199
3: -0.6285393, 0.6923354, -0.9523419, 0.9947283, -1.6232675, 1.6446772
4: -0.3963178, 0.7018194, -0.5820951, 1.0479591, -1.4442769, 1.2839143

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3453781, upper bound: 3.3452929
time: 0.47 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3453781, upper bound: 3.3453012
time: 0.48 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.3456418, 0.4810718, -0.4998576, 0.7077050, -1.0533469, 0.9809293
1: -0.5610024, 0.6868975, -0.8285971, 0.9815257, -1.5425282, 1.5154945
2: -0.3901013, 0.7285742, -0.5478057, 1.1050218, -1.4951231, 1.2763798
3: -0.8672440, 0.8995361, -1.3799070, 1.3170692, -2.1843131, 2.2794433
4: -0.5257008, 0.9459569, -0.8043975, 1.4812887, -2.0069895, 1.7503544

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3210833, upper bound: 3.3508945
time: 0.46 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3196058, upper bound: 3.3441558
time: 0.45 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.2695478, 0.3760729, -0.6188753, 0.9653776, -1.2349253, 0.9949480
1: -0.4326958, 0.5375808, -1.0178568, 1.3381594, -1.7708553, 1.5554374
2: -0.3200988, 0.5548421, -0.6695868, 1.4706283, -1.7907270, 1.2244289
3: -0.6285393, 0.6923354, -1.7465825, 1.7551888, -2.3837280, 2.4389179
4: -0.3963178, 0.7018194, -1.0122970, 1.9336305, -2.3299484, 1.7141165

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3143439, upper bound: 3.3177046
time: 0.43 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3135415, upper bound: 3.3175974
time: 0.46 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.4467238, 0.6215951, -0.3374702, 0.4751466, -0.9218704, 0.9590653
1: -0.7328915, 0.8427354, -0.5469540, 0.6767383, -1.4096296, 1.3896894
2: -0.4924225, 0.9656458, -0.3840603, 0.7139465, -1.2063689, 1.3497061
3: -1.2085794, 1.1289421, -0.8382130, 0.8864570, -2.0950365, 1.9671546
4: -0.7015048, 1.3040695, -0.5186685, 0.9267482, -1.6282527, 1.8227379

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3264811, upper bound: 3.3164643
time: 0.48 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3176822, upper bound: 3.3143772
time: 0.49 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.4117214, 0.5673239, -0.2213027, 0.2906124, -0.7023339, 0.7886266
1: -0.6853573, 0.7768623, -0.3332962, 0.4055558, -1.0909131, 1.1101584
2: -0.4597958, 0.8666953, -0.2738732, 0.4195383, -0.8793340, 1.1405685
3: -1.0987918, 1.0422142, -0.4549357, 0.5165300, -1.6153219, 1.4971498
4: -0.6412443, 1.1624334, -0.2942165, 0.5215645, -1.1628087, 1.4566498

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3175770, upper bound: 3.3141221
time: 0.46 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3175770, upper bound: 3.3155035
time: 0.49 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.4467238, 0.6215951, -0.5934382, 0.9077055, -1.3544292, 1.2150328
1: -0.7328915, 0.8427354, -0.9761899, 1.2567856, -1.9896764, 1.8189247
2: -0.4924225, 0.9656458, -0.6433287, 1.3895657, -1.8819878, 1.6089745
3: -1.2085794, 1.1289421, -1.6682706, 1.6545837, -2.8631630, 2.7972126
4: -0.7015048, 1.3040695, -0.9662716, 1.8339038, -2.5354085, 2.2703412

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3132489, upper bound: 3.3132489
time: 0.44 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3132489, upper bound: 3.3132991
time: 0.46 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.4117214, 0.5673239, -0.4667219, 0.6546656, -1.0663871, 1.0340456
1: -0.6853573, 0.7768623, -0.7764117, 0.9005806, -1.5859379, 1.5532738
2: -0.4597958, 0.8666953, -0.5142344, 1.0202148, -1.4800106, 1.3809297
3: -1.0987918, 1.0422142, -1.2779007, 1.2120466, -2.3108385, 2.3201144
4: -0.6412443, 1.1624334, -0.7455148, 1.3725727, -2.0138168, 1.9079480

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3132645, upper bound: 3.3125399
time: 0.46 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3122852, upper bound: 3.3122852
time: 0.47 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.89 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.89
Output dim: 0, lower bound: -3.2307899, upper bound: 3.1781943
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.89
Output dim: 0, lower bound: -3.2307025, upper bound: 3.1781355
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.89
Output dim: 0, lower bound: -3.2307899, upper bound: 3.1781943
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.89
Output dim: 0, lower bound: -3.2307025, upper bound: 3.1781355
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.89
Output dim: 0, lower bound: -3.3041493, upper bound: 3.3042061
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.89
Output dim: 0, lower bound: -3.3041493, upper bound: 3.3043367
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.89
Output dim: 0, lower bound: -3.3041493, upper bound: 3.3086207
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.89
Output dim: 0, lower bound: -3.3041493, upper bound: 3.3086207
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.89
Output dim: 0, lower bound: -3.3041493, upper bound: 3.3054478
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.89
Output dim: 0, lower bound: -3.3041493, upper bound: 3.3054507
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.89
Output dim: 0, lower bound: -3.3041493, upper bound: 3.3098627
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.89
Output dim: 0, lower bound: -3.3041493, upper bound: 3.3098627
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.89
Output dim: 0, lower bound: -3.3194880, upper bound: 3.3074085
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.89
Output dim: 0, lower bound: -3.3194880, upper bound: 3.3086230
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.89
Output dim: 0, lower bound: -3.3194880, upper bound: 3.3096523
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.89
Output dim: 0, lower bound: -3.3194880, upper bound: 3.3109368
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.89
Output dim: 0, lower bound: -3.3194880, upper bound: 3.3203509
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.89
Output dim: 0, lower bound: -3.3194880, upper bound: 3.3203508
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.89
Output dim: 0, lower bound: -3.3194880, upper bound: 3.3134235
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.89
Output dim: 0, lower bound: -3.3194880, upper bound: 3.3143501
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.89
Output dim: 0, lower bound: -3.2894622, upper bound: 3.2340108
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.89
Output dim: 0, lower bound: -3.2894622, upper bound: 3.3171830
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.89
Output dim: 0, lower bound: -3.2894622, upper bound: 3.2340108
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.89
Output dim: 0, lower bound: -3.2894622, upper bound: 3.3171830
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.89
Output dim: 0, lower bound: -3.3113878, upper bound: 3.2913383
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.89
Output dim: 0, lower bound: -3.3113440, upper bound: 3.3044364
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.89
Output dim: 0, lower bound: -3.3113878, upper bound: 3.2913383
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.89
Output dim: 0, lower bound: -3.3113440, upper bound: 3.3044364
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.89
Output dim: 0, lower bound: -3.3206436, upper bound: 3.3683324
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.89
Output dim: 0, lower bound: -3.3250562, upper bound: 3.3691405
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.89
Output dim: 0, lower bound: -3.3204385, upper bound: 3.3667918
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.89
Output dim: 0, lower bound: -3.3247598, upper bound: 3.3675337
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.89
Output dim: 0, lower bound: -3.2125103, upper bound: 3.2664578
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.89
Output dim: 0, lower bound: -3.2098741, upper bound: 3.2644273
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.89
Output dim: 0, lower bound: -3.2635130, upper bound: 3.3100920
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.89
Output dim: 0, lower bound: -3.3170610, upper bound: 3.3590794
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.89
Output dim: 0, lower bound: -3.2757392, upper bound: 3.2994781
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.89
Output dim: 0, lower bound: -3.2805548, upper bound: 3.3005894
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.89
Output dim: 0, lower bound: -3.2760943, upper bound: 3.2994492
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.89
Output dim: 0, lower bound: -3.3038872, upper bound: 3.3007717
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.89
Output dim: 0, lower bound: -3.2727248, upper bound: 3.2991534
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.89
Output dim: 0, lower bound: -3.2805548, upper bound: 3.3005894
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.89
Output dim: 0, lower bound: -3.2726827, upper bound: 3.2990963
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.89
Output dim: 0, lower bound: -3.2984112, upper bound: 3.3007717
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.89
Output dim: 0, lower bound: -3.3544338, upper bound: 3.3824468
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.89
Output dim: 0, lower bound: -3.3544338, upper bound: 3.3824556
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.89
Output dim: 0, lower bound: -3.3453781, upper bound: 3.3452929
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.89
Output dim: 0, lower bound: -3.3453781, upper bound: 3.3453012
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.89
Output dim: 0, lower bound: -3.3210833, upper bound: 3.3508945
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.89
Output dim: 0, lower bound: -3.3196058, upper bound: 3.3441558
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.89
Output dim: 0, lower bound: -3.3143439, upper bound: 3.3177046
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.89
Output dim: 0, lower bound: -3.3135415, upper bound: 3.3175974
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.89
Output dim: 0, lower bound: -3.3264811, upper bound: 3.3164643
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.89
Output dim: 0, lower bound: -3.3176822, upper bound: 3.3143772
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.89
Output dim: 0, lower bound: -3.3175770, upper bound: 3.3141221
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.89
Output dim: 0, lower bound: -3.3175770, upper bound: 3.3155035
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.89
Output dim: 0, lower bound: -3.3132489, upper bound: 3.3132489
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.89
Output dim: 0, lower bound: -3.3132489, upper bound: 3.3132991
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.89
Output dim: 0, lower bound: -3.3132645, upper bound: 3.3125399
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.89
Output dim: 0, lower bound: -3.3122852, upper bound: 3.3122852

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0667225, 0.0121824, -0.0674421, 0.0174642, -0.0841867, 0.0796245
1: -0.0755771, 0.0255996, -0.0768298, 0.0341327, -0.1097098, 0.1024294
2: -0.0631434, 0.0164710, -0.0640842, 0.0240756, -0.0872190, 0.0805552
3: -0.0689483, 0.0328951, -0.0704525, 0.0419406, -0.1108889, 0.1033476
4: -0.0496707, 0.0219688, -0.0519074, 0.0304504, -0.0801212, 0.0738762

Time for backsubstitution: 2.26 seconds
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=3.5238394737243652
rel_dist={0: [-3.398203369636585, 3.398203369636585]}

## Binary Search with IS_dual_ind Result
status: None
Maximum delta epsilon: None
execution time: 1128.74 seconds
