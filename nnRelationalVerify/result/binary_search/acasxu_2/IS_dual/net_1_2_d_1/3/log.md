## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_2.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 0.5653432899999999


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970)
1: (-0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661)
2: (-0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567)
3: (-0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351)
4: (-0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724)

## BASE Result
execution time: IAR + LP analysis = 1.58 + 1.01 = 2.59 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.5996373, upper bound: 0.5996373


# Binary Search by BASE starts (time budget: 1197.41 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.0909091


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.0909091, mid=0.0909091, abs_max=0.6789970397949219
rel_dist={0: [-0.5944664749150393, 0.5944664749150403]}

## Binary search (step 1) starts
Candidate diff: 0.0454545


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.0454545, mid=0.0454545, abs_max=0.6789970397949219
rel_dist={0: [-0.5819155938898586, 0.581915593889859]}

## Binary search (step 2) starts
Candidate diff: 0.0227273


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0227273, mid=0.0227273, abs_max=0.6789970397949219
rel_dist={0: [-0.5691307785504863, 0.5691307785504858]}

## Binary search (step 3) starts
Candidate diff: 0.0113636


## IAR start
Binary search (step 3): status=Status.VERIFIED, low=0.0113636, high=0.0227273, mid=0.0113636, abs_max=0.6789970397949219
rel_dist={0: [-0.5603966589278331, 0.5603966589278329]}

## Binary search (step 4) starts
Candidate diff: 0.0170455


## IAR start
Binary search (step 4): status=Status.VERIFIED, low=0.0170455, high=0.0227273, mid=0.0170455, abs_max=0.6789970397949219
rel_dist={0: [-0.5653139919093196, 0.5653139919093197]}

## Binary search (step 5) starts
Candidate diff: 0.0198864


## IAR start
Binary search (step 5): status=Status.UNKNOWN, low=0.0170455, high=0.0198864, mid=0.0198864, abs_max=0.6789970397949219
rel_dist={0: [-0.5672873705434701, 0.5672873705434704]}

## Binary search (step 6) starts
Candidate diff: 0.0184659


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0170455, high=0.0184659, mid=0.0184659, abs_max=0.6789970397949219
rel_dist={0: [-0.5663515562000013, 0.5663515562000012]}

## Binary search (step 7) starts
Candidate diff: 0.0177557


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0170455, high=0.0177557, mid=0.0177557, abs_max=0.6789970397949219
rel_dist={0: [-0.5658477313091703, 0.5658477313091701]}

## Binary search (step 8) starts
Candidate diff: 0.0174006


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0170455, high=0.0174006, mid=0.0174006, abs_max=0.6789970397949219
rel_dist={0: [-0.5655834053454235, 0.5655834053454236]}

## Binary search (step 9) starts
Candidate diff: 0.0172230


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0170455, high=0.0172230, mid=0.0172230, abs_max=0.6789970397949219
rel_dist={0: [-0.565448698643106, 0.5654486986431058]}

## Binary search (step 10) starts
Candidate diff: 0.0171342


## IAR start
Binary search (step 10): status=Status.UNKNOWN, low=0.0170455, high=0.0171342, mid=0.0171342, abs_max=0.6789970397949219
rel_dist={0: [-0.5653813459867384, 0.5653813459867383]}

## Binary search (step 11) starts
Candidate diff: 0.0170898


## IAR start
Binary search (step 11): status=Status.UNKNOWN, low=0.0170455, high=0.0170898, mid=0.0170898, abs_max=0.6789970397949219
rel_dist={0: [-0.5653476681722737, 0.565347668172274]}

## Binary search (step 12) starts
Candidate diff: 0.0170676


## IAR start
Binary search (step 12): status=Status.VERIFIED, low=0.0170676, high=0.0170898, mid=0.0170676, abs_max=0.6789970397949219
rel_dist={0: [-0.5653308307370227, 0.5653308307370226]}

## Binary search (step 13) starts
Candidate diff: 0.0170787


## IAR start
Binary search (step 13): status=Status.VERIFIED, low=0.0170787, high=0.0170898, mid=0.0170787, abs_max=0.6789970397949219
rel_dist={0: [-0.5653392488038538, 0.5653392488038538]}

## Binary search (step 14) starts
Candidate diff: 0.0170843


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0170787, high=0.0170843, mid=0.0170843, abs_max=0.6789970397949219
rel_dist={0: [-0.5653434585353336, 0.5653434585353336]}

## Binary search (step 15) starts
Candidate diff: 0.0170815


## IAR start
Binary search (step 15): status=Status.VERIFIED, low=0.0170815, high=0.0170843, mid=0.0170815, abs_max=0.6789970397949219
rel_dist={0: [-0.5653413543928635, 0.5653413543928636]}

## Binary search (step 16) starts
Candidate diff: 0.0170829


## IAR start
Binary search (step 16): status=Status.VERIFIED, low=0.0170829, high=0.0170843, mid=0.0170829, abs_max=0.6789970397949219
rel_dist={0: [-0.5653424072038739, 0.5653424072038737]}

## Binary search (step 17) starts
Candidate diff: 0.0170836


## IAR start
Binary search (step 17): status=Status.VERIFIED, low=0.0170836, high=0.0170843, mid=0.0170836, abs_max=0.6789970397949219
rel_dist={0: [-0.565342932814077, 0.5653429328140769]}

## Binary Search Result
Binary search time: 46.38 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.017083602027241795


# Individual Split (IS_dual) starts
Time budget: 1151.03 seconds

## Binary search (step 0) starts
Candidate diff: 0.0994509


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5628750, upper bound: 0.5950600
time: 0.34 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5885052, upper bound: 0.5885052
time: 0.36 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.84 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 0.84
Output dim: 0, lower bound: -0.5628750, upper bound: 0.5950600
IS_B2, status: Status.UNKNOWN, split count: 1, time: 0.84
Output dim: 0, lower bound: -0.5885052, upper bound: 0.5885052

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -0.0365533, 0.6424438, 0.0674889, 0.4535027, -0.4900560, 0.5749549
1: -0.0992057, 0.8423603, 0.0201817, 0.5976362, -0.6968419, 0.8221787
2: -0.1934414, 0.7124153, -0.0460193, 0.5242411, -0.7176825, 0.7584347
3: -0.2859350, 0.8194001, -0.1192396, 0.5392005, -0.8251356, 0.9386396
4: -0.3152393, 0.9457331, -0.1227688, 0.6923177, -1.0075570, 1.0685018

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B1_B1

### Relational analysis result of IS_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5609737, upper bound: 0.5936294
time: 0.30 seconds

## Relational analysis of IS_B1_B2

### Relational analysis result of IS_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5615310, upper bound: 0.5920751
time: 0.34 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -0.0365533, 0.6424438, -0.0106674, 0.5881721, -0.6247254, 0.6531112
1: -0.0992057, 0.8423603, -0.0680232, 0.7645921, -0.8637978, 0.9103835
2: -0.1934414, 0.7124153, -0.1492940, 0.6635130, -0.8569544, 0.8617094
3: -0.2859350, 0.8194001, -0.2353030, 0.7381678, -1.0241028, 1.0547031
4: -0.3152393, 0.9457331, -0.2566378, 0.8659467, -1.1811860, 1.2023709

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 25

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5885052, upper bound: 0.5628750
time: 0.35 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5885052, upper bound: 0.5885052
time: 0.41 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.34 seconds
IS_B1_B1, status: Status.UNKNOWN, split count: 2, time: 2.34
Output dim: 0, lower bound: -0.5609737, upper bound: 0.5936294
IS_B1_B2, status: Status.UNKNOWN, split count: 2, time: 2.34
Output dim: 0, lower bound: -0.5615310, upper bound: 0.5920751
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 2.34
Output dim: 0, lower bound: -0.5885052, upper bound: 0.5628750
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 2.34
Output dim: 0, lower bound: -0.5885052, upper bound: 0.5885052

## BFS IS instance: IS_B1_B1

### Backsubstitution after applying IS history:
0: -0.0360333, 0.6413742, 0.0667267, 0.5276489, -0.5636822, 0.5746475
1: -0.0985821, 0.8409946, 0.0209925, 0.7202063, -0.8187885, 0.8200021
2: -0.1926503, 0.7112877, -0.0676523, 0.5800259, -0.7726762, 0.7789401
3: -0.2849796, 0.8179674, -0.1276886, 0.6454976, -0.9304772, 0.9456561
4: -0.3141956, 0.9440379, -0.1461909, 0.8046894, -1.1188850, 1.0902288

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_B1_B1_A1

### Relational analysis result of IS_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4865434, upper bound: 0.5794595
time: 0.32 seconds

## Relational analysis of IS_B1_B1_A2

### Relational analysis result of IS_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5608159, upper bound: 0.5935876
time: 0.31 seconds

## BFS IS instance: IS_B1_B2

### Backsubstitution after applying IS history:
0: -0.0365533, 0.6424438, 0.0789177, 0.4454548, -0.4820081, 0.5635260
1: -0.0992057, 0.8423603, 0.0311321, 0.5851125, -0.6843182, 0.8112282
2: -0.1934414, 0.7124153, -0.0367235, 0.5172547, -0.7106961, 0.7491388
3: -0.2859350, 0.8194001, -0.1111584, 0.5267107, -0.8126457, 0.9305584
4: -0.3152393, 0.9457331, -0.1149590, 0.6824017, -0.9976410, 1.0606921

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_B1_B2_A1

### Relational analysis result of IS_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5280490, upper bound: 0.5916821
time: 0.35 seconds

## Relational analysis of IS_B1_B2_A2

### Relational analysis result of IS_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5613670, upper bound: 0.5920423
time: 0.34 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: 0.0674889, 0.4535027, -0.0106674, 0.5881721, -0.5206832, 0.4641702
1: 0.0201817, 0.5976362, -0.0680232, 0.7645921, -0.7444105, 0.6656594
2: -0.0460193, 0.5242411, -0.1492940, 0.6635130, -0.7095323, 0.6735351
3: -0.1192396, 0.5392005, -0.2353030, 0.7381678, -0.8574073, 0.7745036
4: -0.1227688, 0.6923177, -0.2566378, 0.8659467, -0.9887154, 0.9489555

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_A1_A1

### Relational analysis result of IS_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5450628, upper bound: 0.5609737
time: 0.34 seconds

## Relational analysis of IS_B2_A1_A2

### Relational analysis result of IS_B2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5615309, upper bound: 0.5615310
time: 0.35 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -0.0106674, 0.5881721, -0.0106674, 0.5881721, -0.5988396, 0.5988396
1: -0.0680232, 0.7645921, -0.0680232, 0.7645921, -0.8326153, 0.8326153
2: -0.1492940, 0.6635130, -0.1492940, 0.6635130, -0.8128070, 0.8128070
3: -0.2353030, 0.7381678, -0.2353030, 0.7381678, -0.9734708, 0.9734708
4: -0.2566378, 0.8659467, -0.2566378, 0.8659467, -1.1225845, 1.1225845

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B2_A2_B1

### Relational analysis result of IS_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5609737, upper bound: 0.5450628
time: 0.37 seconds

## Relational analysis of IS_B2_A2_B2

### Relational analysis result of IS_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5615310, upper bound: 0.5615309
time: 0.51 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.50 seconds
IS_B1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.50
Output dim: 0, lower bound: -0.4865434, upper bound: 0.5794595
IS_B1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.50
Output dim: 0, lower bound: -0.5608159, upper bound: 0.5935876
IS_B1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.50
Output dim: 0, lower bound: -0.5280490, upper bound: 0.5916821
IS_B1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.50
Output dim: 0, lower bound: -0.5613670, upper bound: 0.5920423
IS_B2_A1_A1, status: Status.VERIFIED, split count: 3, time: 2.50
Output dim: 0, lower bound: -0.5450628, upper bound: 0.5609737
IS_B2_A1_A2, status: Status.VERIFIED, split count: 3, time: 2.50
Output dim: 0, lower bound: -0.5615309, upper bound: 0.5615310
IS_B2_A2_B1, status: Status.VERIFIED, split count: 3, time: 2.50
Output dim: 0, lower bound: -0.5609737, upper bound: 0.5450628
IS_B2_A2_B2, status: Status.VERIFIED, split count: 3, time: 2.50
Output dim: 0, lower bound: -0.5615310, upper bound: 0.5615309

## BFS IS instance: IS_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0580496, 0.6700407, 0.0672615, 0.5269604, -0.5850100, 0.6027792
1: -0.1220827, 0.9192375, 0.0219309, 0.7190909, -0.8411736, 0.8973066
2: -0.2519680, 0.7130269, -0.0668643, 0.5795007, -0.8314688, 0.7798911
3: -0.3335013, 0.8968642, -0.1268568, 0.6444501, -0.9779514, 1.0237210
4: -0.3905666, 1.0050492, -0.1454698, 0.8038490, -1.1944156, 1.1505190

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 24

Time for candidate selection: 4.48 seconds

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_B1_B1_A1_B1

### Relational analysis result of IS_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4812345, upper bound: 0.5765854
time: 0.33 seconds

## Relational analysis of IS_B1_B1_A1_B2

### Relational analysis result of IS_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4564610, upper bound: 0.5717249
time: 0.33 seconds

## BFS IS instance: IS_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0346479, 0.6365551, 0.0667267, 0.5276489, -0.5622969, 0.5698284
1: -0.0968555, 0.8350818, 0.0209925, 0.7202063, -0.8170618, 0.8140893
2: -0.1903567, 0.7059506, -0.0676523, 0.5800259, -0.7703826, 0.7736030
3: -0.2820667, 0.8121664, -0.1276886, 0.6454976, -0.9275643, 0.9398550
4: -0.3110197, 0.9363325, -0.1461909, 0.8046894, -1.1157091, 1.0825233

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B1_B1_A2_A1

### Relational analysis result of IS_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5608159, upper bound: 0.5761132
time: 0.36 seconds

## Relational analysis of IS_B1_B1_A2_A2

### Relational analysis result of IS_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5608159, upper bound: 0.5935876
time: 0.35 seconds

## BFS IS instance: IS_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0585096, 0.6708779, 0.0793476, 0.4448406, -0.5033501, 0.5915303
1: -0.1226516, 0.9202629, 0.0316794, 0.5841250, -0.7067766, 0.8885835
2: -0.2526414, 0.7139475, -0.0360711, 0.5167693, -0.7694107, 0.7500186
3: -0.3343972, 0.8980739, -0.1104538, 0.5257305, -0.8601277, 1.0085278
4: -0.3914803, 1.0063322, -0.1143421, 0.6816227, -1.0731031, 1.1206743

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B1_B2_A1_A1

### Relational analysis result of IS_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5065091, upper bound: 0.5892136
time: 0.34 seconds

## Relational analysis of IS_B1_B2_A1_A2

### Relational analysis result of IS_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5205425, upper bound: 0.5853411
time: 0.35 seconds

## BFS IS instance: IS_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0351349, 0.6375349, 0.0789177, 0.4454548, -0.4805897, 0.5586172
1: -0.0974373, 0.8363609, 0.0311321, 0.5851125, -0.6825498, 0.8052288
2: -0.1911077, 0.7069756, -0.0367235, 0.5172547, -0.7083625, 0.7436991
3: -0.2829589, 0.8134989, -0.1111584, 0.5267107, -0.8096696, 0.9246573
4: -0.3120059, 0.9378967, -0.1149590, 0.6824017, -0.9944075, 1.0528557

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_B2_A2_A1

### Relational analysis result of IS_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5449055, upper bound: 0.5820026
time: 0.36 seconds

## Relational analysis of IS_B1_B2_A2_A2

### Relational analysis result of IS_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5449055, upper bound: 0.5920423
time: 0.35 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.31 seconds
IS_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.31
Output dim: 0, lower bound: -0.4812345, upper bound: 0.5765854
IS_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.31
Output dim: 0, lower bound: -0.4564610, upper bound: 0.5717249
IS_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 2.31
Output dim: 0, lower bound: -0.5608159, upper bound: 0.5761132
IS_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 2.31
Output dim: 0, lower bound: -0.5608159, upper bound: 0.5935876
IS_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 2.31
Output dim: 0, lower bound: -0.5065091, upper bound: 0.5892136
IS_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 2.31
Output dim: 0, lower bound: -0.5205425, upper bound: 0.5853411
IS_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 2.31
Output dim: 0, lower bound: -0.5449055, upper bound: 0.5820026
IS_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 2.31
Output dim: 0, lower bound: -0.5449055, upper bound: 0.5920423

## BFS IS instance: IS_B1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0580496, 0.6700407, 0.0739131, 0.4990853, -0.5571349, 0.5961276
1: -0.1220827, 0.9192375, 0.0326638, 0.6669300, -0.7890127, 0.8865737
2: -0.2519680, 0.7130269, -0.0474222, 0.5638554, -0.8158234, 0.7604491
3: -0.3335013, 0.8968642, -0.1147792, 0.6014205, -0.9349219, 1.0116434
4: -0.3905666, 1.0050492, -0.1279743, 0.7612551, -1.1518217, 1.1330235

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 5

## Relational analysis of IS_B1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_B1_B1_A1_B1_A1

### Relational analysis result of IS_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4801153, upper bound: 0.5666637
time: 0.33 seconds

## Relational analysis of IS_B1_B1_A1_B1_A2

### Relational analysis result of IS_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4776906, upper bound: 0.5743023
time: 0.33 seconds

## BFS IS instance: IS_B1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0580496, 0.6700407, 0.0709836, 0.5033181, -0.5613676, 0.5990570
1: -0.1220827, 0.9192375, 0.0264117, 0.6710327, -0.7931154, 0.8928258
2: -0.2519680, 0.7130269, -0.0490327, 0.5692213, -0.8211893, 0.7620596
3: -0.3335013, 0.8968642, -0.1084092, 0.6068593, -0.9403607, 1.0052733
4: -0.3905666, 1.0050492, -0.1227696, 0.7654014, -1.1559680, 1.1278188

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 5

## Relational analysis of IS_B1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_B1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_B1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_B1_B1_A1_B2_B1

### Relational analysis result of IS_B1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4529460, upper bound: 0.5701191
time: 0.36 seconds

## Relational analysis of IS_B1_B1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_B1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_B1_B1_A1_B2_A1

### Relational analysis result of IS_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4544027, upper bound: 0.5698546
time: 0.33 seconds

## Relational analysis of IS_B1_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 5
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 24
type: A, layer: 5, pos: 27
type: A, layer: 5, pos: 22
type: B, layer: 5, pos: 22
type: A, layer: 5, pos: 24
type: B, layer: 5, pos: 27
type: B, layer: 5, pos: 32
type: A, layer: 5, pos: 49
type: A, layer: 5, pos: 32
type: A, layer: 5, pos: 17
type: A, layer: 5, pos: 18
type: B, layer: 5, pos: 18
type: A, layer: 5, pos: 11
type: A, layer: 5, pos: 9
type: B, layer: 5, pos: 35
type: B, layer: 5, pos: 11
type: B, layer: 5, pos: 9
type: A, layer: 5, pos: 35

Time for candidate selection: 3.17 seconds

### Candidate
type: A, layer: 5, pos: 19

## Relational analysis of IS_B1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 19

## Relational analysis of IS_B1_B1_A1_B2_B1

### Relational analysis result of IS_B1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4503213, upper bound: 0.5656143
time: 0.33 seconds

## Relational analysis of IS_B1_B1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 24

## Relational analysis of IS_B1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 27

## Relational analysis of IS_B1_B1_A1_B2_A1

### Relational analysis result of IS_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4538708, upper bound: 0.5684859
time: 0.33 seconds

## Relational analysis of IS_B1_B1_A1_B2_A2

### Relational analysis result of IS_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4541527, upper bound: 0.5684653
time: 0.31 seconds

## BFS IS instance: IS_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: 0.0681343, 0.4512217, 0.0667267, 0.5276489, -0.4595146, 0.3844950
1: 0.0211298, 0.5938053, 0.0209925, 0.7202063, -0.6990765, 0.5728127
2: -0.0444727, 0.5225873, -0.0676523, 0.5800259, -0.6244986, 0.5902396
3: -0.1180182, 0.5358654, -0.1276886, 0.6454976, -0.7635158, 0.6635540
4: -0.1213741, 0.6891791, -0.1461909, 0.8046894, -0.9260635, 0.8353699

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 25

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_B1_A2_A1_A1

### Relational analysis result of IS_B1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5460763, upper bound: 0.5761132
time: 0.42 seconds

## Relational analysis of IS_B1_B1_A2_A1_A2

### Relational analysis result of IS_B1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5460763, upper bound: 0.5761132
time: 0.36 seconds

## BFS IS instance: IS_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0084354, 0.5821132, 0.0667267, 0.5276489, -0.5360843, 0.5153865
1: -0.0654132, 0.7562742, 0.0209925, 0.7202063, -0.7856196, 0.7352817
2: -0.1457217, 0.6569647, -0.0676523, 0.5800259, -0.7257476, 0.7246171
3: -0.2306992, 0.7298000, -0.1276886, 0.6454976, -0.8761968, 0.8574886
4: -0.2514850, 0.8565888, -0.1461909, 0.8046894, -1.0561744, 1.0027797

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_B1_A2_A2_A1

### Relational analysis result of IS_B1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5460763, upper bound: 0.5841992
time: 0.33 seconds

## Relational analysis of IS_B1_B1_A2_A2_A2

### Relational analysis result of IS_B1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5460763, upper bound: 0.5935876
time: 0.36 seconds

## BFS IS instance: IS_B1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0395788, 0.6407486, 0.0793476, 0.4448406, -0.4844193, 0.5614010
1: -0.1016229, 0.8804287, 0.0316794, 0.5841250, -0.6857479, 0.8487493
2: -0.2235353, 0.6828082, -0.0360711, 0.5167693, -0.7403046, 0.7188792
3: -0.3008456, 0.8524711, -0.1104538, 0.5257305, -0.8265761, 0.9629249
4: -0.3538617, 0.9610786, -0.1143421, 0.6816227, -1.0354844, 1.0754207

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_B1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_B1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_B1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_B1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_B2_A1_A1_B1

### Relational analysis result of IS_B1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5065091, upper bound: 0.5802624
time: 0.35 seconds

## Relational analysis of IS_B1_B2_A1_A1_B2

### Relational analysis result of IS_B1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5065091, upper bound: 0.5853411
time: 0.33 seconds

## BFS IS instance: IS_B1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0587145, 0.6864064, 0.0793476, 0.4448406, -0.5035551, 0.6070588
1: -0.1260673, 0.9511679, 0.0316794, 0.5841250, -0.7101923, 0.9194885
2: -0.2601123, 0.7187865, -0.0360711, 0.5167693, -0.7768816, 0.7548576
3: -0.3402169, 0.9311209, -0.1104538, 0.5257305, -0.8659474, 1.0415747
4: -0.4028527, 1.0292013, -0.1143421, 0.6816227, -1.0844754, 1.1435434

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_B1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_B2_A1_A2_B1

### Relational analysis result of IS_B1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5205425, upper bound: 0.5802624
time: 0.33 seconds

## Relational analysis of IS_B1_B2_A1_A2_B2

### Relational analysis result of IS_B1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5205425, upper bound: 0.5853411
time: 0.33 seconds

## BFS IS instance: IS_B1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0277573, 0.7424477, 0.0789177, 0.4454548, -0.4732121, 0.6635300
1: -0.0962460, 1.0040389, 0.0311321, 0.5851125, -0.6813585, 0.9729068
2: -0.2100114, 0.7920854, -0.0367235, 0.5172547, -0.7272661, 0.8288089
3: -0.2813025, 0.9581012, -0.1111584, 0.5267107, -0.8080132, 1.0692596
4: -0.3271902, 1.0964527, -0.1149590, 0.6824017, -1.0095918, 1.2114117

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B1_B2_A2_A1_A1

### Relational analysis result of IS_B1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5449055, upper bound: 0.5739167
time: 0.35 seconds

## Relational analysis of IS_B1_B2_A2_A1_A2

### Relational analysis result of IS_B1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5449055, upper bound: 0.5820026
time: 0.38 seconds

## BFS IS instance: IS_B1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0131834, 0.6111942, 0.0789177, 0.4454548, -0.4586382, 0.5322765
1: -0.0753851, 0.8058444, 0.0311321, 0.5851125, -0.6604976, 0.7747123
2: -0.1671922, 0.6776322, -0.0367235, 0.5172547, -0.6844469, 0.7143557
3: -0.2535679, 0.7779697, -0.1111584, 0.5267107, -0.7802786, 0.8891280
4: -0.2808745, 0.9000369, -0.1149590, 0.6824017, -0.9632761, 1.0149959

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B1_B2_A2_A2_A1

### Relational analysis result of IS_B1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5449055, upper bound: 0.5739036
time: 0.38 seconds

## Relational analysis of IS_B1_B2_A2_A2_A2

### Relational analysis result of IS_B1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5449055, upper bound: 0.5920423
time: 0.36 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.41 seconds
IS_B1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.41
Output dim: 0, lower bound: -0.4801153, upper bound: 0.5666637
IS_B1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.41
Output dim: 0, lower bound: -0.4776906, upper bound: 0.5743023
IS_B1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.41
Output dim: 0, lower bound: -0.4538708, upper bound: 0.5684859
IS_B1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.41
Output dim: 0, lower bound: -0.4541527, upper bound: 0.5684653
IS_B1_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 2.41
Output dim: 0, lower bound: -0.5460763, upper bound: 0.5761132
IS_B1_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 2.41
Output dim: 0, lower bound: -0.5460763, upper bound: 0.5761132
IS_B1_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 2.41
Output dim: 0, lower bound: -0.5460763, upper bound: 0.5841992
IS_B1_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 2.41
Output dim: 0, lower bound: -0.5460763, upper bound: 0.5935876
IS_B1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.41
Output dim: 0, lower bound: -0.5065091, upper bound: 0.5802624
IS_B1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.41
Output dim: 0, lower bound: -0.5065091, upper bound: 0.5853411
IS_B1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.41
Output dim: 0, lower bound: -0.5205425, upper bound: 0.5802624
IS_B1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.41
Output dim: 0, lower bound: -0.5205425, upper bound: 0.5853411
IS_B1_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 2.41
Output dim: 0, lower bound: -0.5449055, upper bound: 0.5739167
IS_B1_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 2.41
Output dim: 0, lower bound: -0.5449055, upper bound: 0.5820026
IS_B1_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 2.41
Output dim: 0, lower bound: -0.5449055, upper bound: 0.5739036
IS_B1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 2.41
Output dim: 0, lower bound: -0.5449055, upper bound: 0.5920423

## BFS IS instance: IS_B1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0162394, 0.5857634, 0.0739131, 0.4990853, -0.5153247, 0.5118503
1: -0.0739384, 0.7863106, 0.0326638, 0.6669300, -0.7408683, 0.7536468
2: -0.1851189, 0.6332189, -0.0474222, 0.5638554, -0.7489743, 0.6806411
3: -0.2635653, 0.7544556, -0.1147792, 0.6014205, -0.8649858, 0.8692349
4: -0.2944105, 0.8652927, -0.1279743, 0.7612551, -1.0556656, 0.9932670

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 5

## Relational analysis of IS_B1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_B1_B1_A1_B1_A1_B1

### Relational analysis result of IS_B1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4801153, upper bound: 0.5666637
time: 0.38 seconds

## Relational analysis of IS_B1_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_B1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_B1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_B1_B1_A1_B1_A1_A1

### Relational analysis result of IS_B1_B1_A1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4782461, upper bound: 0.5647600
time: 0.33 seconds

## Relational analysis of IS_B1_B1_A1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_B1_B1_A1_B1_A1_B1

### Relational analysis result of IS_B1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4608566, upper bound: 0.5666637
time: 0.33 seconds

## Relational analysis of IS_B1_B1_A1_B1_A1_B2

### Relational analysis result of IS_B1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4608566, upper bound: 0.5666637
time: 0.38 seconds

## BFS IS instance: IS_B1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0367553, 0.6442507, 0.0739131, 0.4990853, -0.5358406, 0.5703376
1: -0.0946175, 0.8780697, 0.0326638, 0.6669300, -0.7615474, 0.8454059
2: -0.2165998, 0.6898979, -0.0474222, 0.5638554, -0.7804552, 0.7373201
3: -0.2930150, 0.8517221, -0.1147792, 0.6014205, -0.8944355, 0.9665014
4: -0.3485820, 0.9636061, -0.1279743, 0.7612551, -1.1098372, 1.0915804

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_B1_B1_A1_B1_A2_B1

### Relational analysis result of IS_B1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4776906, upper bound: 0.5743023
time: 0.34 seconds

## Relational analysis of IS_B1_B1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 5

## Relational analysis of IS_B1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_B1_B1_A1_B1_A2_A1

### Relational analysis result of IS_B1_B1_A1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4699730, upper bound: 0.5648873
time: 0.45 seconds

## Relational analysis of IS_B1_B1_A1_B1_A2_A2

### Relational analysis result of IS_B1_B1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4699730, upper bound: 0.5743023
time: 0.32 seconds

## BFS IS instance: IS_B1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0464455, 0.6526178, 0.0709836, 0.5033181, -0.5497636, 0.5816342
1: -0.1098495, 0.9000089, 0.0264117, 0.6710327, -0.7808822, 0.8735972
2: -0.2358221, 0.6951980, -0.0490327, 0.5692213, -0.8050434, 0.7442307
3: -0.3121704, 0.8722180, -0.1084092, 0.6068593, -0.9190298, 0.9806272
4: -0.3670949, 0.9796290, -0.1227696, 0.7654014, -1.1324962, 1.1023986

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 5
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 24
type: A, layer: 5, pos: 22
type: B, layer: 5, pos: 22
type: A, layer: 5, pos: 24
type: A, layer: 5, pos: 49
type: B, layer: 5, pos: 32
type: A, layer: 5, pos: 32
type: A, layer: 5, pos: 17
type: A, layer: 5, pos: 18
type: B, layer: 5, pos: 18
type: A, layer: 5, pos: 9
type: A, layer: 5, pos: 11
type: B, layer: 5, pos: 35
type: B, layer: 5, pos: 11
type: B, layer: 5, pos: 27
type: B, layer: 5, pos: 9
type: A, layer: 5, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 5, pos: 19

## Relational analysis of IS_B1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 19

## Relational analysis of IS_B1_B1_A1_B2_A1_B1

### Relational analysis result of IS_B1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4476746, upper bound: 0.5623338
time: 0.34 seconds

## Relational analysis of IS_B1_B1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 24

## Relational analysis of IS_B1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 22

## Relational analysis of IS_B1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 22

## Relational analysis of IS_B1_B1_A1_B2_A1_B1

### Relational analysis result of IS_B1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4534345, upper bound: 0.5681975
time: 0.34 seconds

## Relational analysis of IS_B1_B1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 24

## Relational analysis of IS_B1_B1_A1_B2_A1_A1

### Relational analysis result of IS_B1_B1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4466653, upper bound: 0.5661825
time: 0.34 seconds

## Relational analysis of IS_B1_B1_A1_B2_A1_A2

### Relational analysis result of IS_B1_B1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4495857, upper bound: 0.5656353
time: 0.33 seconds

## BFS IS instance: IS_B1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0414629, 0.6190577, 0.0709836, 0.5033181, -0.5447810, 0.5480741
1: -0.1036322, 0.8099380, 0.0264117, 0.6710327, -0.7746649, 0.7835264
2: -0.2019371, 0.6892880, -0.0490327, 0.5692213, -0.7711584, 0.7383206
3: -0.2992053, 0.8078072, -0.1084092, 0.6068593, -0.9060646, 0.9162164
4: -0.3411905, 0.9143052, -0.1227696, 0.7654014, -1.1065918, 1.0370748

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 5
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 24
type: A, layer: 5, pos: 22
type: B, layer: 5, pos: 22
type: A, layer: 5, pos: 49
type: A, layer: 5, pos: 24
type: A, layer: 5, pos: 18
type: B, layer: 5, pos: 32
type: A, layer: 5, pos: 32
type: A, layer: 5, pos: 17
type: B, layer: 5, pos: 18
type: A, layer: 5, pos: 9
type: A, layer: 5, pos: 11
type: B, layer: 5, pos: 11
type: B, layer: 5, pos: 27
type: B, layer: 5, pos: 35
type: A, layer: 5, pos: 35
type: B, layer: 5, pos: 9

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 5, pos: 19

## Relational analysis of IS_B1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 19

## Relational analysis of IS_B1_B1_A1_B2_A2_B1

### Relational analysis result of IS_B1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4482060, upper bound: 0.5623131
time: 0.36 seconds

## Relational analysis of IS_B1_B1_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 24

## Relational analysis of IS_B1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 22

## Relational analysis of IS_B1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 22

## Relational analysis of IS_B1_B1_A1_B2_A2_B1

### Relational analysis result of IS_B1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4536546, upper bound: 0.5681766
time: 0.34 seconds

## Relational analysis of IS_B1_B1_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 49

## Relational analysis of IS_B1_B1_A1_B2_A2_A1

### Relational analysis result of IS_B1_B1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4540519, upper bound: 0.5683354
time: 0.34 seconds

## Relational analysis of IS_B1_B1_A1_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 24

## Relational analysis of IS_B1_B1_A1_B2_A2_A1

### Relational analysis result of IS_B1_B1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4466248, upper bound: 0.5661630
time: 0.39 seconds

## Relational analysis of IS_B1_B1_A1_B2_A2_A2

### Relational analysis result of IS_B1_B1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4498737, upper bound: 0.5657465
time: 0.31 seconds

## BFS IS instance: IS_B1_B1_A2_A1_A1

### Backsubstitution after applying IS history:
0: 0.0671313, 0.5257025, 0.0667267, 0.5276489, -0.4605176, 0.4589758
1: 0.0216136, 0.7168880, 0.0209925, 0.7202063, -0.6985927, 0.6958954
2: -0.0664191, 0.5788780, -0.0676523, 0.5800259, -0.6464450, 0.6465304
3: -0.1267802, 0.6426113, -0.1276886, 0.6454976, -0.7722778, 0.7703000
4: -0.1452075, 0.8021576, -0.1461909, 0.8046894, -0.9498969, 0.9483485

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_B1_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_B1_B1_A2_A1_A1_A1

### Relational analysis result of IS_B1_B1_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5473323, upper bound: 0.5754648
time: 0.36 seconds

## Relational analysis of IS_B1_B1_A2_A1_A1_A2

### Relational analysis result of IS_B1_B1_A2_A1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5219667, upper bound: 0.4998705
time: 0.34 seconds

## BFS IS instance: IS_B1_B1_A2_A1_A2

### Backsubstitution after applying IS history:
0: 0.0793607, 0.4436836, 0.0667267, 0.5276489, -0.4482882, 0.3769569
1: 0.0318375, 0.5821407, 0.0209925, 0.7202063, -0.6883689, 0.5611482
2: -0.0354755, 0.5160056, -0.0676523, 0.5800259, -0.6155014, 0.5836579
3: -0.1102905, 0.5241475, -0.1276886, 0.6454976, -0.7557881, 0.6518361
4: -0.1139328, 0.6799669, -0.1461909, 0.8046894, -0.9186223, 0.8261578

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_B1_B1_A2_A1_A2_B1

### Relational analysis result of IS_B1_B1_A2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5493908, upper bound: 0.5001832
time: 0.36 seconds

## Relational analysis of IS_B1_B1_A2_A1_A2_B2

### Relational analysis result of IS_B1_B1_A2_A1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5219667, upper bound: 0.4998705
time: 0.36 seconds

## BFS IS instance: IS_B1_B1_A2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0054744, 0.7100763, 0.0667267, 0.5276489, -0.5331234, 0.6433496
1: -0.0693125, 0.9573513, 0.0209925, 0.7202063, -0.7895188, 0.9363588
2: -0.1748363, 0.7621485, -0.0676523, 0.5800259, -0.7548622, 0.8298008
3: -0.2408400, 0.9047788, -0.1276886, 0.6454976, -0.8863376, 1.0324674
4: -0.2802821, 1.0476310, -0.1461909, 0.8046894, -1.0849715, 1.1938219

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_B1_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_B1_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B1_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B1_B1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B1_B1_A2_A2_A1_A1

### Relational analysis result of IS_B1_B1_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5075200, upper bound: 0.5841175
time: 0.34 seconds

## Relational analysis of IS_B1_B1_A2_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_B1_B1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_B1_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_B1_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_B1_B1_A2_A2_A1_A1

### Relational analysis result of IS_B1_B1_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5185566, upper bound: 0.5836719
time: 0.40 seconds

## Relational analysis of IS_B1_B1_A2_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_B1_B1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 39

Time for candidate selection: 4.80 seconds

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_B1_B1_A2_A2_A1_A1

### Relational analysis result of IS_B1_B1_A2_A2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5454151, upper bound: 0.5614828
time: 0.36 seconds

## Relational analysis of IS_B1_B1_A2_A2_A1_A2

### Relational analysis result of IS_B1_B1_A2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5335370, upper bound: 0.5743467
time: 0.40 seconds

## BFS IS instance: IS_B1_B1_A2_A2_A2

### Backsubstitution after applying IS history:
0: 0.0100145, 0.5674599, 0.0667267, 0.5276489, -0.5176344, 0.5007333
1: -0.0486783, 0.7368349, 0.0209925, 0.7202063, -0.7688846, 0.7158424
2: -0.1268167, 0.6409237, -0.0676523, 0.5800259, -0.7068427, 0.7085760
3: -0.2082636, 0.7057808, -0.1276886, 0.6454976, -0.8537612, 0.8334695
4: -0.2279943, 0.8359103, -0.1461909, 0.8046894, -1.0326837, 0.9821012

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_B1_B1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_B1_B1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_B1_B1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B1_B1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_B1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B1_B1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B1_B1_A2_A2_A2_A1

### Relational analysis result of IS_B1_B1_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5075200, upper bound: 0.5935060
time: 0.36 seconds

## Relational analysis of IS_B1_B1_A2_A2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_B1_B1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_B1_B1_A2_A2_A2_A1

### Relational analysis result of IS_B1_B1_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5185566, upper bound: 0.5929929
time: 0.37 seconds

## Relational analysis of IS_B1_B1_A2_A2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_B1_B1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_B1_B1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 5

Time for candidate selection: 4.83 seconds

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_B1_B1_A2_A2_A2_A1

### Relational analysis result of IS_B1_B1_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5454151, upper bound: 0.5735434
time: 0.37 seconds

## Relational analysis of IS_B1_B1_A2_A2_A2_A2

### Relational analysis result of IS_B1_B1_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5335370, upper bound: 0.5837352
time: 0.37 seconds

## BFS IS instance: IS_B1_B2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0395788, 0.6407486, 0.0894709, 0.4175571, -0.4571359, 0.5512778
1: -0.1016229, 0.8804287, 0.0457507, 0.5343553, -0.6359782, 0.8346780
2: -0.2235353, 0.6828082, -0.0096244, 0.5019960, -0.7255313, 0.6924325
3: -0.3008456, 0.8524711, -0.0870266, 0.4841275, -0.7849731, 0.9394976
4: -0.3538617, 0.9610786, -0.0887296, 0.6444156, -0.9982773, 1.0498083

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B1_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_B1_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_B1_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_B1_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_B1_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B1_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B1_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_B1_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_B1_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_B1_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 24

Time for candidate selection: 4.05 seconds

### Candidate
type: B, layer: 3, pos: 5

## Relational analysis of IS_B1_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_B1_B2_A1_A1_B1_B1

### Relational analysis result of IS_B1_B2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5017936, upper bound: 0.5836195
time: 0.35 seconds

## Relational analysis of IS_B1_B2_A1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_B1_B2_A1_A1_B1_B1

### Relational analysis result of IS_B1_B2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4865264, upper bound: 0.5859924
time: 0.34 seconds

## Relational analysis of IS_B1_B2_A1_A1_B1_B2

### Relational analysis result of IS_B1_B2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5008990, upper bound: 0.5762508
time: 0.35 seconds

## BFS IS instance: IS_B1_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0395788, 0.6407486, 0.0805085, 0.4525800, -0.4921588, 0.5602401
1: -0.1016229, 0.8804287, 0.0280254, 0.5985373, -0.7001602, 0.8524033
2: -0.2235353, 0.6828082, -0.0423912, 0.5218675, -0.7454028, 0.7251993
3: -0.3008456, 0.8524711, -0.1158811, 0.5381635, -0.8390091, 0.9683521
4: -0.3538617, 0.9610786, -0.1197159, 0.6938162, -1.0476779, 1.0807946

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B1_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_B1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_B1_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_B1_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_B1_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B1_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_B1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_B1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_B1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 24

Time for candidate selection: 4.06 seconds

### Candidate
type: B, layer: 3, pos: 41

## Relational analysis of IS_B1_B2_A1_A1_B2_B1

### Relational analysis result of IS_B1_B2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5065091, upper bound: 0.5892136
time: 0.35 seconds

## Relational analysis of IS_B1_B2_A1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 5

## Relational analysis of IS_B1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_B1_B2_A1_A1_B2_B1

### Relational analysis result of IS_B1_B2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4865264, upper bound: 0.5877100
time: 0.34 seconds

## Relational analysis of IS_B1_B2_A1_A1_B2_B2

### Relational analysis result of IS_B1_B2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5008990, upper bound: 0.5808138
time: 0.34 seconds

## BFS IS instance: IS_B1_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0587145, 0.6864064, 0.0894709, 0.4175571, -0.4762716, 0.5969355
1: -0.1260673, 0.9511679, 0.0457507, 0.5343553, -0.6604226, 0.9054172
2: -0.2601123, 0.7187865, -0.0096244, 0.5019960, -0.7621083, 0.7284109
3: -0.3402169, 0.9311209, -0.0870266, 0.4841275, -0.8243443, 1.0181475
4: -0.4028527, 1.0292013, -0.0887296, 0.6444156, -1.0472683, 1.1179309

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_B1_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B1_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_B1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_B1_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B1_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_B1_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_B1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_B1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_B1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 24

Time for candidate selection: 3.98 seconds

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of IS_B1_B2_A1_A2_B1_A1

### Relational analysis result of IS_B1_B2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5205425, upper bound: 0.5802624
time: 0.39 seconds

## Relational analysis of IS_B1_B2_A1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 5

## Relational analysis of IS_B1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_B1_B2_A1_A2_B1_B1

### Relational analysis result of IS_B1_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5030079, upper bound: 0.5726087
time: 0.36 seconds

## Relational analysis of IS_B1_B2_A1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_B1_B2_A1_A2_B1_B1

### Relational analysis result of IS_B1_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4999460, upper bound: 0.5792840
time: 0.35 seconds

## Relational analysis of IS_B1_B2_A1_A2_B1_B2

### Relational analysis result of IS_B1_B2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5143186, upper bound: 0.5696791
time: 0.38 seconds

## BFS IS instance: IS_B1_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0587145, 0.6864064, 0.0805085, 0.4525800, -0.5112945, 0.6058979
1: -0.1260673, 0.9511679, 0.0280254, 0.5985373, -0.7246045, 0.9231426
2: -0.2601123, 0.7187865, -0.0423912, 0.5218675, -0.7819798, 0.7611777
3: -0.3402169, 0.9311209, -0.1158811, 0.5381635, -0.8783804, 1.0470021
4: -0.4028527, 1.0292013, -0.1197159, 0.6938162, -1.0966688, 1.1489172

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B1_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_B1_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_B1_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_B1_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B1_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_B1_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_B1_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B1_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_B1_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_B1_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 24

Time for candidate selection: 4.02 seconds

### Candidate
type: B, layer: 3, pos: 41

## Relational analysis of IS_B1_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of IS_B1_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 5

## Relational analysis of IS_B1_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_B1_B2_A1_A2_B2_B1

### Relational analysis result of IS_B1_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4999461, upper bound: 0.5794951
time: 0.37 seconds

## Relational analysis of IS_B1_B2_A1_A2_B2_B2

### Relational analysis result of IS_B1_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5143186, upper bound: 0.5702970
time: 0.38 seconds

## BFS IS instance: IS_B1_B2_A2_A1_A1

### Backsubstitution after applying IS history:
0: 0.0644008, 0.5302621, 0.0789177, 0.4454548, -0.3810540, 0.4513444
1: 0.0192177, 0.7222201, 0.0311321, 0.5851125, -0.5658948, 0.6910880
2: -0.0694884, 0.5835138, -0.0367235, 0.5172547, -0.5867431, 0.6202373
3: -0.1317451, 0.6470909, -0.1111584, 0.5267107, -0.6584558, 0.7582493
4: -0.1483382, 0.8069319, -0.1149590, 0.6824017, -0.8307399, 0.9218909

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_B1_B2_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_B1_B2_A2_A1_A1_A1

### Relational analysis result of IS_B1_B2_A2_A1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4691153, upper bound: 0.5493909
time: 0.35 seconds

## Relational analysis of IS_B1_B2_A2_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_B1_B2_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B1_B2_A2_A1_A1_A1

### Relational analysis result of IS_B1_B2_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4677842, upper bound: 0.5749386
time: 0.37 seconds

## Relational analysis of IS_B1_B2_A2_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B1_B2_A2_A1_A1_A1

### Relational analysis result of IS_B1_B2_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5302056, upper bound: 0.5691786
time: 0.36 seconds

## Relational analysis of IS_B1_B2_A2_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_B1_B2_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B1_B2_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_B2_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_B1_B2_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_B1_B2_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_B1_B2_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 5

Time for candidate selection: 5.11 seconds

### Candidate
type: B, layer: 3, pos: 5

## Relational analysis of IS_B1_B2_A2_A1_A1_B1

### Relational analysis result of IS_B1_B2_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4652818, upper bound: 0.5749012
time: 0.37 seconds

## Relational analysis of IS_B1_B2_A2_A1_A1_B2

### Relational analysis result of IS_B1_B2_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5448622, upper bound: 0.5744227
time: 0.40 seconds

## BFS IS instance: IS_B1_B2_A2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0054744, 0.7100763, 0.0789177, 0.4454548, -0.4509293, 0.6311586
1: -0.0693125, 0.9573513, 0.0311321, 0.5851125, -0.6544250, 0.9262192
2: -0.1748363, 0.7621485, -0.0367235, 0.5172547, -0.6920910, 0.7988720
3: -0.2408400, 0.9047788, -0.1111584, 0.5267107, -0.7675507, 1.0159371
4: -0.2802821, 1.0476310, -0.1149590, 0.6824017, -0.9626838, 1.1625900

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_B1_B2_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_B1_B2_A2_A1_A2_A1

### Relational analysis result of IS_B1_B2_A2_A1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4691153, upper bound: 0.5563270
time: 0.35 seconds

## Relational analysis of IS_B1_B2_A2_A1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B1_B2_A2_A1_A2_A1

### Relational analysis result of IS_B1_B2_A2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5302056, upper bound: 0.5760937
time: 0.36 seconds

## Relational analysis of IS_B1_B2_A2_A1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_B1_B2_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_B1_B2_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_B2_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B1_B2_A2_A1_A2_A1

### Relational analysis result of IS_B1_B2_A2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4677842, upper bound: 0.5818537
time: 0.41 seconds

## Relational analysis of IS_B1_B2_A2_A1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B1_B2_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_B1_B2_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_B1_B2_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_B1_B2_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 39

Time for candidate selection: 5.17 seconds

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_B1_B2_A2_A1_A2_B1

### Relational analysis result of IS_B1_B2_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5397815, upper bound: 0.5788453
time: 0.40 seconds

## Relational analysis of IS_B1_B2_A2_A1_A2_B2

### Relational analysis result of IS_B1_B2_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5425539, upper bound: 0.5690052
time: 0.40 seconds

## BFS IS instance: IS_B1_B2_A2_A2_A1

### Backsubstitution after applying IS history:
0: 0.0793607, 0.4436836, 0.0789177, 0.4454548, -0.3660941, 0.3647659
1: 0.0318375, 0.5821407, 0.0311321, 0.5851125, -0.5532750, 0.5510086
2: -0.0354755, 0.5160056, -0.0367235, 0.5172547, -0.5527302, 0.5527291
3: -0.1102905, 0.5241475, -0.1111584, 0.5267107, -0.6370012, 0.6353058
4: -0.1139328, 0.6799669, -0.1149590, 0.6824017, -0.7963345, 0.7949259

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_B2_A2_A2_A1_B1

### Relational analysis result of IS_B1_B2_A2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5607108, upper bound: 0.5567627
time: 0.40 seconds

## Relational analysis of IS_B1_B2_A2_A2_A1_B2

### Relational analysis result of IS_B1_B2_A2_A2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5506120, upper bound: 0.5542917
time: 0.36 seconds

## BFS IS instance: IS_B1_B2_A2_A2_A2

### Backsubstitution after applying IS history:
0: 0.0100145, 0.5674599, 0.0789177, 0.4454548, -0.4354403, 0.4885422
1: -0.0486783, 0.7368349, 0.0311321, 0.5851125, -0.6337908, 0.7057028
2: -0.1268167, 0.6409237, -0.0367235, 0.5172547, -0.6440715, 0.6776472
3: -0.2082636, 0.7057808, -0.1111584, 0.5267107, -0.7349743, 0.8169392
4: -0.2279943, 0.8359103, -0.1149590, 0.6824017, -0.9103960, 0.9508693

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_B1_B2_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_B1_B2_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B1_B2_A2_A2_A2_A1

### Relational analysis result of IS_B1_B2_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5410317, upper bound: 0.5892508
time: 0.35 seconds

## Relational analysis of IS_B1_B2_A2_A2_A2_A2

### Relational analysis result of IS_B1_B2_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5506118, upper bound: 0.5853646
time: 0.42 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.13 seconds
IS_B1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.4608566, upper bound: 0.5666637
IS_B1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.4608566, upper bound: 0.5666637
IS_B1_B1_A1_B1_A2_A1, status: Status.VERIFIED, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.4699730, upper bound: 0.5648873
IS_B1_B1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.4699730, upper bound: 0.5743023
IS_B1_B1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.4466653, upper bound: 0.5661825
IS_B1_B1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.4495857, upper bound: 0.5656353
IS_B1_B1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.4466248, upper bound: 0.5661630
IS_B1_B1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.4498737, upper bound: 0.5657465
IS_B1_B1_A2_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.5473323, upper bound: 0.5754648
IS_B1_B1_A2_A1_A1_A2, status: Status.VERIFIED, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.5219667, upper bound: 0.4998705
IS_B1_B1_A2_A1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.5493908, upper bound: 0.5001832
IS_B1_B1_A2_A1_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.5219667, upper bound: 0.4998705
IS_B1_B1_A2_A2_A1_A1, status: Status.VERIFIED, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.5454151, upper bound: 0.5614828
IS_B1_B1_A2_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.5335370, upper bound: 0.5743467
IS_B1_B1_A2_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.5454151, upper bound: 0.5735434
IS_B1_B1_A2_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.5335370, upper bound: 0.5837352
IS_B1_B2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.4865264, upper bound: 0.5859924
IS_B1_B2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.5008990, upper bound: 0.5762508
IS_B1_B2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.4865264, upper bound: 0.5877100
IS_B1_B2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.5008990, upper bound: 0.5808138
IS_B1_B2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.4999460, upper bound: 0.5792840
IS_B1_B2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.5143186, upper bound: 0.5696791
IS_B1_B2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.4999461, upper bound: 0.5794951
IS_B1_B2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.5143186, upper bound: 0.5702970
IS_B1_B2_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.4652818, upper bound: 0.5749012
IS_B1_B2_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.5448622, upper bound: 0.5744227
IS_B1_B2_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.5397815, upper bound: 0.5788453
IS_B1_B2_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.5425539, upper bound: 0.5690052
IS_B1_B2_A2_A2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.5607108, upper bound: 0.5567627
IS_B1_B2_A2_A2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.5506120, upper bound: 0.5542917
IS_B1_B2_A2_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.5410317, upper bound: 0.5892508
IS_B1_B2_A2_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.5506118, upper bound: 0.5853646

## BFS IS instance: IS_B1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0162394, 0.5857634, 0.0840328, 0.4702755, -0.4865149, 0.5017306
1: -0.0739384, 0.7863106, 0.0484066, 0.6074100, -0.6813483, 0.7379040
2: -0.1851189, 0.6332189, -0.0199761, 0.5499380, -0.7350569, 0.6531951
3: -0.2635653, 0.7544556, -0.0935962, 0.5514519, -0.8150172, 0.8480518
4: -0.2944105, 0.8652927, -0.1050725, 0.7053619, -0.9997724, 0.9703652

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_B1_B1_A1_B1_A1_B1_B1

### Relational analysis result of IS_B1_B1_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4632020, upper bound: 0.5666637
time: 0.35 seconds

## Relational analysis of IS_B1_B1_A1_B1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 5

## Relational analysis of IS_B1_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_B1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_B1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4632020, upper bound: 0.5666637
time: 0.33 seconds

## Relational analysis of IS_B1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_B1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4608566, upper bound: 0.5666637
time: 0.32 seconds

## BFS IS instance: IS_B1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0162394, 0.5857634, 0.0805278, 0.4908124, -0.5070518, 0.5052356
1: -0.0739384, 0.7863106, 0.0433629, 0.6536850, -0.7276233, 0.7429478
2: -0.1851189, 0.6332189, -0.0363598, 0.5577300, -0.7428489, 0.6695787
3: -0.2635653, 0.7544556, -0.1024306, 0.5891042, -0.8526695, 0.8568862
4: -0.2944105, 0.8652927, -0.1189392, 0.7504232, -1.0448337, 0.9842319

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 5

## Relational analysis of IS_B1_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_B1_B1_A1_B1_A1_B2_B1

### Relational analysis result of IS_B1_B1_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4632020, upper bound: 0.5666637
time: 0.35 seconds

## Relational analysis of IS_B1_B1_A1_B1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_B1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_B1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4801153, upper bound: 0.5666637
time: 0.35 seconds

## Relational analysis of IS_B1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_B1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4776906, upper bound: 0.5666637
time: 0.34 seconds

## BFS IS instance: IS_B1_B1_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0239775, 0.6403115, 0.0739131, 0.4990853, -0.5230628, 0.5663984
1: -0.0847045, 0.8598368, 0.0326638, 0.6669300, -0.7516345, 0.8271730
2: -0.1902319, 0.6998742, -0.0474222, 0.5638554, -0.7540873, 0.7472964
3: -0.2780581, 0.8351519, -0.1147792, 0.6014205, -0.8794786, 0.9499311
4: -0.3221620, 0.9590466, -0.1279743, 0.7612551, -1.0834172, 1.0870209

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 5

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 5
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 24
type: B, layer: 5, pos: 22
type: A, layer: 5, pos: 22
type: A, layer: 5, pos: 27
type: A, layer: 5, pos: 32
type: A, layer: 5, pos: 49
type: B, layer: 5, pos: 24
type: B, layer: 5, pos: 27
type: B, layer: 5, pos: 32
type: A, layer: 5, pos: 18
type: B, layer: 5, pos: 18
type: A, layer: 5, pos: 9
type: B, layer: 5, pos: 9
type: A, layer: 5, pos: 4
type: A, layer: 5, pos: 11
type: B, layer: 5, pos: 35
type: B, layer: 5, pos: 11
type: A, layer: 5, pos: 17

Time for candidate selection: 2.60 seconds

### Candidate
type: A, layer: 5, pos: 19

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 19

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 24

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 22

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 22

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 27

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 32

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 49

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 24

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 27

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 32

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 18

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 18

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 9

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 9

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 4

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 11

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 35

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 11

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 17

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 7
type: A, layer: 7, pos: 5
type: B, layer: 7, pos: 5
type: A, layer: 7, pos: 36
type: B, layer: 7, pos: 36
type: A, layer: 7, pos: 47
type: A, layer: 7, pos: 37
type: A, layer: 7, pos: 42
type: B, layer: 7, pos: 47
type: B, layer: 7, pos: 37
type: A, layer: 7, pos: 35
type: B, layer: 7, pos: 35
type: A, layer: 7, pos: 11
type: B, layer: 7, pos: 11
type: A, layer: 7, pos: 49
type: B, layer: 7, pos: 49
type: A, layer: 7, pos: 26
type: B, layer: 7, pos: 42
type: A, layer: 7, pos: 19
type: B, layer: 7, pos: 19
type: A, layer: 7, pos: 23
type: B, layer: 7, pos: 13
type: A, layer: 7, pos: 13
type: B, layer: 7, pos: 23
type: B, layer: 7, pos: 26
type: A, layer: 7, pos: 8

Time for candidate selection: 10.72 seconds

### Candidate
type: A, layer: 7, pos: 5

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 5

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 7, pos: 36

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 36

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 7, pos: 47

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 7, pos: 37

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 7, pos: 42

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 47

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 37

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 7, pos: 35

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 35

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 7, pos: 11

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 11

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 7, pos: 49

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 49

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 7, pos: 26

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 42

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 7, pos: 19

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 19

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 7, pos: 23

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 13

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 7, pos: 13

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 23

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 26

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 7, pos: 8

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 9
type: A, layer: 9, pos: 43
type: B, layer: 9, pos: 43
type: A, layer: 9, pos: 45
type: A, layer: 9, pos: 3
type: B, layer: 9, pos: 45
type: B, layer: 9, pos: 3
type: A, layer: 9, pos: 42
type: B, layer: 9, pos: 42
type: A, layer: 9, pos: 35
type: A, layer: 9, pos: 38
type: B, layer: 9, pos: 35
type: A, layer: 9, pos: 39
type: A, layer: 9, pos: 22
type: B, layer: 9, pos: 22
type: A, layer: 9, pos: 28
type: B, layer: 9, pos: 9
type: A, layer: 9, pos: 9
type: B, layer: 9, pos: 39
type: B, layer: 9, pos: 28
type: A, layer: 9, pos: 14
type: A, layer: 9, pos: 30
type: B, layer: 9, pos: 30
type: B, layer: 9, pos: 14
type: A, layer: 9, pos: 11
type: A, layer: 9, pos: 48
type: A, layer: 9, pos: 7
type: A, layer: 9, pos: 12
type: A, layer: 9, pos: 29
type: A, layer: 9, pos: 18
type: B, layer: 9, pos: 11
type: B, layer: 9, pos: 48
type: A, layer: 9, pos: 25
type: B, layer: 9, pos: 25
type: B, layer: 9, pos: 29
type: A, layer: 9, pos: 36
type: B, layer: 9, pos: 12
type: A, layer: 9, pos: 34
type: A, layer: 9, pos: 37
type: B, layer: 9, pos: 18
type: A, layer: 9, pos: 21
type: A, layer: 9, pos: 41
type: A, layer: 9, pos: 26
type: A, layer: 9, pos: 40
type: A, layer: 9, pos: 24
type: B, layer: 9, pos: 40
type: B, layer: 9, pos: 37

Time for candidate selection: 20.60 seconds

### Candidate
type: A, layer: 9, pos: 43

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 43

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 45

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 3

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 45

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 3

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 42

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 42

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 35

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 38

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 35

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 39

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 22

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 22

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 28

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 9

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 9

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 39

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 28

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 14

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 30

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 30

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 14

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 11

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 48

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 7

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 12

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 29

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 18

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 11

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 48

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 25

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 25

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 29

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 36

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 12

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 34

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 37

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 18

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 21

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 41

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 26

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 40

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 24

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 40

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 37

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 11
type: A, layer: 11, pos: 20
type: B, layer: 11, pos: 20
type: A, layer: 11, pos: 8
type: B, layer: 11, pos: 8
type: A, layer: 11, pos: 12
type: A, layer: 11, pos: 39
type: B, layer: 11, pos: 12
type: B, layer: 11, pos: 39
type: A, layer: 11, pos: 27
type: A, layer: 11, pos: 38
type: B, layer: 11, pos: 38
type: B, layer: 11, pos: 27
type: A, layer: 11, pos: 21
type: A, layer: 11, pos: 7
type: A, layer: 11, pos: 6
type: A, layer: 11, pos: 35
type: B, layer: 11, pos: 21
type: B, layer: 11, pos: 7
type: B, layer: 11, pos: 6
type: B, layer: 11, pos: 35
type: A, layer: 11, pos: 17
type: A, layer: 11, pos: 10
type: A, layer: 11, pos: 13
type: B, layer: 11, pos: 17
type: A, layer: 11, pos: 0
type: B, layer: 11, pos: 0
type: A, layer: 11, pos: 26
type: A, layer: 11, pos: 3
type: B, layer: 11, pos: 10
type: A, layer: 11, pos: 33
type: A, layer: 11, pos: 28
type: A, layer: 11, pos: 15
type: B, layer: 11, pos: 26
type: B, layer: 11, pos: 3
type: A, layer: 11, pos: 31
type: B, layer: 11, pos: 15
type: A, layer: 11, pos: 23
type: B, layer: 11, pos: 28
type: A, layer: 11, pos: 18
type: B, layer: 11, pos: 31
type: B, layer: 11, pos: 23
type: A, layer: 11, pos: 44
type: A, layer: 11, pos: 47
type: B, layer: 11, pos: 44
type: A, layer: 11, pos: 36
type: A, layer: 11, pos: 2
type: B, layer: 11, pos: 2
type: A, layer: 11, pos: 45
type: B, layer: 11, pos: 36
type: A, layer: 11, pos: 37
type: B, layer: 11, pos: 37
type: A, layer: 11, pos: 48
type: A, layer: 11, pos: 34
type: A, layer: 11, pos: 22
type: A, layer: 11, pos: 19
type: A, layer: 11, pos: 42
type: A, layer: 11, pos: 49
type: B, layer: 11, pos: 49
type: A, layer: 11, pos: 43
type: A, layer: 11, pos: 4
type: B, layer: 11, pos: 43
type: A, layer: 11, pos: 5
type: A, layer: 11, pos: 11

Time for candidate selection: 39.01 seconds

### Candidate
type: A, layer: 11, pos: 20

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 11, pos: 20

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 8

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 11, pos: 8

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 12

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 39

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 11, pos: 12

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 11, pos: 39

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 27

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 38

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 11, pos: 38

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 11, pos: 27

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 21

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 7

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 6

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 35

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 11, pos: 21

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 11, pos: 7

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 11, pos: 6

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 11, pos: 35

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 17

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 10

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 13

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 11, pos: 17

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 0

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 11, pos: 0

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 26

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 3

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 11, pos: 10

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 33

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 28

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 15

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 11, pos: 26

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 11, pos: 3

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 31

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 11, pos: 15

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 23

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 11, pos: 28

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 18

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 11, pos: 31

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 11, pos: 23

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 44

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 47

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 11, pos: 44

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 36

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 2

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 11, pos: 2

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 45

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 11, pos: 36

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 37

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 11, pos: 37

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 48

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 34

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 22

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 19

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 42

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 49

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 11, pos: 49

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 43

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 4

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 11, pos: 43

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 5

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 11

## Relational analysis of IS_B1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

No IS candidates found
Binary search (step 0): status=Status.UNKNOWN, low=0.0170836, high=0.0994509, mid=0.0994509, abs_max=0.6789970397949219
rel_dist={0: [-0.5950600062778149, 0.5950600062778155]}

## Binary search (step 1) starts
Candidate diff: 0.0582672


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5854215, upper bound: 0.5584218
time: 0.36 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5837545, upper bound: 0.5837545
time: 0.32 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.84 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.84
Output dim: 0, lower bound: -0.5854215, upper bound: 0.5584218
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.84
Output dim: 0, lower bound: -0.5837545, upper bound: 0.5837545

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.0674889, 0.4535027, -0.0339178, 0.6323221, -0.5648332, 0.4874205
1: 0.0201817, 0.5976362, -0.0957146, 0.8296235, -0.8094419, 0.6933507
2: -0.0460193, 0.5242411, -0.1887267, 0.7016845, -0.7477039, 0.7129678
3: -0.1192396, 0.5392005, -0.2802914, 0.8066562, -0.9258958, 0.8194920
4: -0.1227688, 0.6923177, -0.3088930, 0.9303064, -1.0530752, 1.0012107

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5801078, upper bound: 0.5559137
time: 0.35 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5834923, upper bound: 0.5566878
time: 0.34 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0106674, 0.5881721, -0.0365533, 0.6424438, -0.6531112, 0.6247254
1: -0.0680232, 0.7645921, -0.0992057, 0.8423603, -0.9103835, 0.8637978
2: -0.1492940, 0.6635130, -0.1934414, 0.7124153, -0.8617094, 0.8569544
3: -0.2353030, 0.7381678, -0.2859350, 0.8194001, -1.0547031, 1.0241028
4: -0.2566378, 0.8659467, -0.3152393, 0.9457331, -1.2023709, 1.1811860

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5584218, upper bound: 0.5837545
time: 0.34 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5584218, upper bound: 0.5837545
time: 0.35 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.31 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 2.31
Output dim: 0, lower bound: -0.5801078, upper bound: 0.5559137
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 2.31
Output dim: 0, lower bound: -0.5834923, upper bound: 0.5566878
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.31
Output dim: 0, lower bound: -0.5584218, upper bound: 0.5837545
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.31
Output dim: 0, lower bound: -0.5584218, upper bound: 0.5837545

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: 0.0667267, 0.5276489, -0.0303714, 0.6251736, -0.5584469, 0.5580204
1: 0.0209925, 0.7202063, -0.0916682, 0.8202824, -0.7992899, 0.8118746
2: -0.0676523, 0.5800259, -0.1832590, 0.6942219, -0.7618742, 0.7632849
3: -0.1276886, 0.6454976, -0.2737818, 0.7969659, -0.9246545, 0.9192794
4: -0.1461909, 0.8046894, -0.3017198, 0.9190533, -1.0652442, 1.1064092

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5741290, upper bound: 0.5559137
time: 0.34 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5741290, upper bound: 0.5559137
time: 0.35 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: 0.0789177, 0.4454548, -0.0339178, 0.6323221, -0.5534044, 0.4793726
1: 0.0311321, 0.5851125, -0.0957146, 0.8296235, -0.7984914, 0.6808271
2: -0.0367235, 0.5172547, -0.1887267, 0.7016845, -0.7384080, 0.7059814
3: -0.1111584, 0.5267107, -0.2802914, 0.8066562, -0.9178146, 0.8070021
4: -0.1149590, 0.6824017, -0.3088930, 0.9303064, -1.0452654, 0.9912946

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5833949, upper bound: 0.5200946
time: 0.40 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5834019, upper bound: 0.5565219
time: 0.36 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0106674, 0.5881721, 0.0674889, 0.4535027, -0.4641702, 0.5206832
1: -0.0680232, 0.7645921, 0.0201817, 0.5976362, -0.6656594, 0.7444105
2: -0.1492940, 0.6635130, -0.0460193, 0.5242411, -0.6735351, 0.7095323
3: -0.2353030, 0.7381678, -0.1192396, 0.5392005, -0.7745036, 0.8574073
4: -0.2566378, 0.8659467, -0.1227688, 0.6923177, -0.9489555, 0.9887154

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5559137, upper bound: 0.5620327
time: 0.36 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5566878, upper bound: 0.5820731
time: 0.34 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0106674, 0.5881721, -0.0106674, 0.5881721, -0.5988396, 0.5988396
1: -0.0680232, 0.7645921, -0.0680232, 0.7645921, -0.8326153, 0.8326153
2: -0.1492940, 0.6635130, -0.1492940, 0.6635130, -0.8128070, 0.8128070
3: -0.2353030, 0.7381678, -0.2353030, 0.7381678, -0.9734708, 0.9734708
4: -0.2566378, 0.8659467, -0.2566378, 0.8659467, -1.1225845, 1.1225845

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5450628, upper bound: 0.5640388
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5566877, upper bound: 0.5820733
time: 0.38 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.46 seconds
IS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 0, lower bound: -0.5741290, upper bound: 0.5559137
IS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 0, lower bound: -0.5741290, upper bound: 0.5559137
IS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 0, lower bound: -0.5833949, upper bound: 0.5200946
IS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 0, lower bound: -0.5834019, upper bound: 0.5565219
IS_A2_B1_B1, status: Status.VERIFIED, split count: 3, time: 2.46
Output dim: 0, lower bound: -0.5559137, upper bound: 0.5620327
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 0, lower bound: -0.5566878, upper bound: 0.5820731
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 2.46
Output dim: 0, lower bound: -0.5450628, upper bound: 0.5640388
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 0, lower bound: -0.5566877, upper bound: 0.5820733

## BFS IS instance: IS_A1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0667267, 0.5276489, 0.0688059, 0.4503864, -0.3836597, 0.4588430
1: 0.0209925, 0.7202063, 0.0221454, 0.5923078, -0.5713153, 0.6980609
2: -0.0676523, 0.5800259, -0.0432372, 0.5219170, -0.5895693, 0.6232631
3: -0.1276886, 0.6454976, -0.1169564, 0.5344946, -0.6621833, 0.7624540
4: -0.1461909, 0.8046894, -0.1202323, 0.6880264, -0.8342173, 0.9249218

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_B1

### Relational analysis result of IS_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5731324, upper bound: 0.5462336
time: 0.33 seconds

## Relational analysis of IS_A1_A1_B1_B2

### Relational analysis result of IS_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5731324, upper bound: 0.5462336
time: 0.34 seconds

## BFS IS instance: IS_A1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0667267, 0.5276489, -0.0069911, 0.5818021, -0.5150754, 0.5346401
1: 0.0209925, 0.7202063, -0.0639071, 0.7555361, -0.7345436, 0.7841135
2: -0.0676523, 0.5800259, -0.1438155, 0.6568471, -0.7244994, 0.7238414
3: -0.1276886, 0.6454976, -0.2285144, 0.7286912, -0.8563798, 0.8740121
4: -0.1461909, 0.8046894, -0.2494051, 0.8558180, -1.0020089, 1.0540946

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B2_B1

### Relational analysis result of IS_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5731324, upper bound: 0.5462336
time: 0.37 seconds

## Relational analysis of IS_A1_A1_B2_B2

### Relational analysis result of IS_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5731324, upper bound: 0.5462336
time: 0.33 seconds

## BFS IS instance: IS_A1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0798948, 0.4440666, -0.0560553, 0.6618679, -0.5819731, 0.5001220
1: 0.0323762, 0.5828813, -0.1193147, 0.9090872, -0.8767111, 0.7021960
2: -0.0352423, 0.5161585, -0.2481698, 0.7042297, -0.7394720, 0.7643282
3: -0.1095568, 0.5244954, -0.3287466, 0.8861656, -0.9957224, 0.8532420
4: -0.1135570, 0.6806400, -0.3854021, 0.9929305, -1.1064875, 1.0660421

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_A2_B1_B1

### Relational analysis result of IS_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5817954, upper bound: 0.5063967
time: 0.33 seconds

## Relational analysis of IS_A1_A2_B1_B2

### Relational analysis result of IS_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5818159, upper bound: 0.5187152
time: 0.35 seconds

## BFS IS instance: IS_A1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0789177, 0.4454548, -0.0326071, 0.6279494, -0.5490317, 0.4780619
1: 0.0311321, 0.5851125, -0.0942498, 0.8241325, -0.7930004, 0.6793624
2: -0.0367235, 0.5172547, -0.1865596, 0.6968601, -0.7335836, 0.7038143
3: -0.1111584, 0.5267107, -0.2775359, 0.8013141, -0.9124725, 0.8042466
4: -0.1149590, 0.6824017, -0.3058759, 0.9233649, -1.0383239, 0.9882776

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A2_B2_B1

### Relational analysis result of IS_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5729922, upper bound: 0.5449055
time: 0.37 seconds

## Relational analysis of IS_A1_A2_B2_B2

### Relational analysis result of IS_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5729922, upper bound: 0.5451149
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0106674, 0.5881721, 0.0789177, 0.4454548, -0.4561223, 0.5092544
1: -0.0680232, 0.7645921, 0.0311321, 0.5851125, -0.6531357, 0.7334600
2: -0.1492940, 0.6635130, -0.0367235, 0.5172547, -0.6665487, 0.7002365
3: -0.2353030, 0.7381678, -0.1111584, 0.5267107, -0.7620137, 0.8493261
4: -0.2566378, 0.8659467, -0.1149590, 0.6824017, -0.9390395, 0.9809057

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5450628, upper bound: 0.5730985
time: 0.35 seconds

## Relational analysis of IS_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5450628, upper bound: 0.5730985
time: 0.35 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0083361, 0.5719668, -0.0106674, 0.5881721, -0.5798361, 0.5826342
1: -0.0506135, 0.7433118, -0.0680232, 0.7645921, -0.8152056, 0.8113350
2: -0.1297052, 0.6455543, -0.1492940, 0.6635130, -0.7932182, 0.7948483
3: -0.2118729, 0.7126579, -0.2353030, 0.7381678, -0.9500406, 0.9479610
4: -0.2322460, 0.8428500, -0.2566378, 0.8659467, -1.0981927, 1.0994878

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5561863, upper bound: 0.5620327
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5561863, upper bound: 0.5620327
time: 0.38 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.43 seconds
IS_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 2.43
Output dim: 0, lower bound: -0.5731324, upper bound: 0.5462336
IS_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 2.43
Output dim: 0, lower bound: -0.5731324, upper bound: 0.5462336
IS_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.43
Output dim: 0, lower bound: -0.5731324, upper bound: 0.5462336
IS_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.43
Output dim: 0, lower bound: -0.5731324, upper bound: 0.5462336
IS_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 2.43
Output dim: 0, lower bound: -0.5817954, upper bound: 0.5063967
IS_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 2.43
Output dim: 0, lower bound: -0.5818159, upper bound: 0.5187152
IS_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.43
Output dim: 0, lower bound: -0.5729922, upper bound: 0.5449055
IS_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.43
Output dim: 0, lower bound: -0.5729922, upper bound: 0.5451149
IS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.43
Output dim: 0, lower bound: -0.5450628, upper bound: 0.5730985
IS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.43
Output dim: 0, lower bound: -0.5450628, upper bound: 0.5730985
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 2.43
Output dim: 0, lower bound: -0.5561863, upper bound: 0.5620327
IS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 2.43
Output dim: 0, lower bound: -0.5561863, upper bound: 0.5620327

## BFS IS instance: IS_A1_A1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0667267, 0.5276489, 0.0667906, 0.5272058, -0.4604791, 0.4608583
1: 0.0209925, 0.7202063, 0.0211356, 0.7194211, -0.6984286, 0.6990708
2: -0.0676523, 0.5800259, -0.0674331, 0.5798492, -0.6475016, 0.6474590
3: -0.1276886, 0.6454976, -0.1275190, 0.6448306, -0.7725192, 0.7730166
4: -0.1461909, 0.8046894, -0.1461201, 0.8042681, -0.9504590, 0.9508095

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A1_B1_B1_A1

### Relational analysis result of IS_A1_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5002262, upper bound: 0.5455672
time: 0.35 seconds

## Relational analysis of IS_A1_A1_B1_B1_A2

### Relational analysis result of IS_A1_A1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4999297, upper bound: 0.5219670
time: 0.33 seconds

## BFS IS instance: IS_A1_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0667267, 0.5276489, 0.0789177, 0.4454548, -0.3787282, 0.4487312
1: 0.0209925, 0.7202063, 0.0311321, 0.5851125, -0.5641200, 0.6890742
2: -0.0676523, 0.5800259, -0.0367235, 0.5172547, -0.5849071, 0.6167494
3: -0.1276886, 0.6454976, -0.1111584, 0.5267107, -0.6543993, 0.7566560
4: -0.1461909, 0.8046894, -0.1149590, 0.6824017, -0.8285925, 0.9196484

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A1_B1_B2_A1

### Relational analysis result of IS_A1_A1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5002262, upper bound: 0.5455672
time: 0.35 seconds

## Relational analysis of IS_A1_A1_B1_B2_A2

### Relational analysis result of IS_A1_A1_B1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4999297, upper bound: 0.5219670
time: 0.34 seconds

## BFS IS instance: IS_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0667267, 0.5276489, -0.0069308, 0.7141405, -0.6474138, 0.5345798
1: 0.0209925, 0.7202063, -0.0711197, 0.9631418, -0.9421493, 0.7913260
2: -0.0676523, 0.5800259, -0.1773483, 0.7663486, -0.8340009, 0.7573742
3: -0.1276886, 0.6454976, -0.2441539, 0.9109904, -1.0386790, 0.8896515
4: -0.1461909, 0.8046894, -0.2841340, 1.0538890, -1.2000799, 1.0888234

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_A1_B2_B1_B1

### Relational analysis result of IS_A1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5698856, upper bound: 0.5076778
time: 0.34 seconds

## Relational analysis of IS_A1_A1_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A1_B2_B1_B1

### Relational analysis result of IS_A1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5703186, upper bound: 0.5187193
time: 0.34 seconds

## Relational analysis of IS_A1_A1_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 39

Time for candidate selection: 4.90 seconds

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_A1_B2_B1_B1

### Relational analysis result of IS_A1_A1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5510566, upper bound: 0.5455418
time: 0.35 seconds

## Relational analysis of IS_A1_A1_B2_B1_B2

### Relational analysis result of IS_A1_A1_B2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5633369, upper bound: 0.5337426
time: 0.38 seconds

## BFS IS instance: IS_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0667267, 0.5276489, 0.0083361, 0.5719668, -0.5052401, 0.5193129
1: 0.0209925, 0.7202063, -0.0506135, 0.7433118, -0.7223193, 0.7708198
2: -0.0676523, 0.5800259, -0.1297052, 0.6455543, -0.7132066, 0.7097311
3: -0.1276886, 0.6454976, -0.2118729, 0.7126579, -0.8403466, 0.8573705
4: -0.1461909, 0.8046894, -0.2322460, 0.8428500, -0.9890409, 1.0369354

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_A1_B2_B2_B1

### Relational analysis result of IS_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5698856, upper bound: 0.5157469
time: 0.34 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A1_B2_B2_B1

### Relational analysis result of IS_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5703186, upper bound: 0.5266457
time: 0.34 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 5

Time for candidate selection: 4.98 seconds

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_A1_B2_B2_B1

### Relational analysis result of IS_A1_A1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5510566, upper bound: 0.5455418
time: 0.37 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2

### Relational analysis result of IS_A1_A1_B2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5633369, upper bound: 0.5337426
time: 0.38 seconds

## BFS IS instance: IS_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0798948, 0.4440666, -0.0371455, 0.6319925, -0.5520977, 0.4812121
1: 0.0323762, 0.5828813, -0.0984493, 0.8694278, -0.8370516, 0.6813307
2: -0.0352423, 0.5161585, -0.2191198, 0.6732124, -0.7084547, 0.7352782
3: -0.1095568, 0.5244954, -0.2955074, 0.8407961, -0.9503528, 0.8200029
4: -0.1135570, 0.6806400, -0.3479116, 0.9478350, -1.0613919, 1.0285516

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_A2_B1_B1_A1

### Relational analysis result of IS_A1_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5332210, upper bound: 0.5063908
time: 0.36 seconds

## Relational analysis of IS_A1_A2_B1_B1_A2

### Relational analysis result of IS_A1_A2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5332210, upper bound: 0.5063967
time: 0.35 seconds

## BFS IS instance: IS_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0807357, 0.4424569, -0.0564591, 0.6783893, -0.5976536, 0.4989160
1: 0.0329686, 0.5799750, -0.1230820, 0.9407243, -0.9077557, 0.7030571
2: -0.0340396, 0.5152770, -0.2560430, 0.7099727, -0.7440123, 0.7713200
3: -0.1087598, 0.5220919, -0.3352190, 0.9201487, -1.0289085, 0.8573109
4: -0.1124970, 0.6784713, -0.3973327, 1.0164365, -1.1289334, 1.0758040

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_A2_B1_B2_A1

### Relational analysis result of IS_A1_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5332210, upper bound: 0.5076913
time: 0.34 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2

### Relational analysis result of IS_A1_A2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5332210, upper bound: 0.5187153
time: 0.34 seconds

## BFS IS instance: IS_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0789177, 0.4454548, -0.0253866, 0.7327344, -0.6538167, 0.4708414
1: 0.0311321, 0.5851125, -0.0928965, 0.9913819, -0.9602498, 0.6780090
2: -0.0367235, 0.5172547, -0.2056899, 0.7820052, -0.8187287, 0.7229446
3: -0.1111584, 0.5267107, -0.2749252, 0.9446915, -1.0558498, 0.8016359
4: -0.1149590, 0.6824017, -0.3207877, 1.0814164, -1.1963754, 1.0031893

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_A2_B2_B1_B1

### Relational analysis result of IS_A1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5710482, upper bound: 0.5449055
time: 0.36 seconds

## Relational analysis of IS_A1_A2_B2_B1_B2

### Relational analysis result of IS_A1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5710482, upper bound: 0.5449055
time: 0.39 seconds

## BFS IS instance: IS_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0789177, 0.4454548, -0.0105512, 0.6016853, -0.5227676, 0.4560061
1: 0.0311321, 0.5851125, -0.0722103, 0.7936555, -0.7625234, 0.6573228
2: -0.0367235, 0.5172547, -0.1624763, 0.6675156, -0.7042391, 0.6797310
3: -0.1111584, 0.5267107, -0.2479746, 0.7656057, -0.8767641, 0.7746853
4: -0.1149590, 0.6824017, -0.2745736, 0.8855274, -1.0004864, 0.9569752

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_A2_B2_B2_B1

### Relational analysis result of IS_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5710482, upper bound: 0.5451149
time: 0.37 seconds

## Relational analysis of IS_A1_A2_B2_B2_B2

### Relational analysis result of IS_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5710482, upper bound: 0.5451149
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0069308, 0.7141405, 0.0789177, 0.4454548, -0.4523857, 0.6352228
1: -0.0711197, 0.9631418, 0.0311321, 0.5851125, -0.6562322, 0.9320097
2: -0.1773483, 0.7663486, -0.0367235, 0.5172547, -0.6946030, 0.8030721
3: -0.2441539, 0.9109904, -0.1111584, 0.5267107, -0.7708645, 1.0221487
4: -0.2841340, 1.0538890, -0.1149590, 0.6824017, -0.9665357, 1.1688480

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_B2_A1_A1

### Relational analysis result of IS_A2_B1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4692780, upper bound: 0.5415272
time: 0.37 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_B2_A1_A1

### Relational analysis result of IS_A2_B1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4679259, upper bound: 0.5604052
time: 0.38 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 39

Time for candidate selection: 5.12 seconds

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_B2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5341666, upper bound: 0.5697477
time: 0.35 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5426809, upper bound: 0.5656389
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0083361, 0.5719668, 0.0789177, 0.4454548, -0.4371188, 0.4930490
1: -0.0506135, 0.7433118, 0.0311321, 0.5851125, -0.6357260, 0.7121797
2: -0.1297052, 0.6455543, -0.0367235, 0.5172547, -0.6469599, 0.6822778
3: -0.2118729, 0.7126579, -0.1111584, 0.5267107, -0.7385836, 0.8238163
4: -0.2322460, 0.8428500, -0.1149590, 0.6824017, -0.9146477, 0.9578090

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_B2_A2_A1

### Relational analysis result of IS_A2_B1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4692780, upper bound: 0.5494218
time: 0.36 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_B2_A2_A1

### Relational analysis result of IS_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4679259, upper bound: 0.5750863
time: 0.34 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 5

Time for candidate selection: 5.11 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_B2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5341666, upper bound: 0.5817325
time: 0.39 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5426809, upper bound: 0.5733778
time: 0.39 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 7.88 seconds
IS_A1_A1_B1_B1_A1, status: Status.VERIFIED, split count: 5, time: 7.88
Output dim: 0, lower bound: -0.5002262, upper bound: 0.5455672
IS_A1_A1_B1_B1_A2, status: Status.VERIFIED, split count: 5, time: 7.88
Output dim: 0, lower bound: -0.4999297, upper bound: 0.5219670
IS_A1_A1_B1_B2_A1, status: Status.VERIFIED, split count: 5, time: 7.88
Output dim: 0, lower bound: -0.5002262, upper bound: 0.5455672
IS_A1_A1_B1_B2_A2, status: Status.VERIFIED, split count: 5, time: 7.88
Output dim: 0, lower bound: -0.4999297, upper bound: 0.5219670
IS_A1_A1_B2_B1_B1, status: Status.VERIFIED, split count: 5, time: 7.88
Output dim: 0, lower bound: -0.5510566, upper bound: 0.5455418
IS_A1_A1_B2_B1_B2, status: Status.VERIFIED, split count: 5, time: 7.88
Output dim: 0, lower bound: -0.5633369, upper bound: 0.5337426
IS_A1_A1_B2_B2_B1, status: Status.VERIFIED, split count: 5, time: 7.88
Output dim: 0, lower bound: -0.5510566, upper bound: 0.5455418
IS_A1_A1_B2_B2_B2, status: Status.VERIFIED, split count: 5, time: 7.88
Output dim: 0, lower bound: -0.5633369, upper bound: 0.5337426
IS_A1_A2_B1_B1_A1, status: Status.VERIFIED, split count: 5, time: 7.88
Output dim: 0, lower bound: -0.5332210, upper bound: 0.5063908
IS_A1_A2_B1_B1_A2, status: Status.VERIFIED, split count: 5, time: 7.88
Output dim: 0, lower bound: -0.5332210, upper bound: 0.5063967
IS_A1_A2_B1_B2_A1, status: Status.VERIFIED, split count: 5, time: 7.88
Output dim: 0, lower bound: -0.5332210, upper bound: 0.5076913
IS_A1_A2_B1_B2_A2, status: Status.VERIFIED, split count: 5, time: 7.88
Output dim: 0, lower bound: -0.5332210, upper bound: 0.5187153
IS_A1_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 7.88
Output dim: 0, lower bound: -0.5710482, upper bound: 0.5449055
IS_A1_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 7.88
Output dim: 0, lower bound: -0.5710482, upper bound: 0.5449055
IS_A1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 7.88
Output dim: 0, lower bound: -0.5710482, upper bound: 0.5451149
IS_A1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 7.88
Output dim: 0, lower bound: -0.5710482, upper bound: 0.5451149
IS_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 7.88
Output dim: 0, lower bound: -0.5341666, upper bound: 0.5697477
IS_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 7.88
Output dim: 0, lower bound: -0.5426809, upper bound: 0.5656389
IS_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 7.88
Output dim: 0, lower bound: -0.5341666, upper bound: 0.5817325
IS_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 7.88
Output dim: 0, lower bound: -0.5426809, upper bound: 0.5733778

## BFS IS instance: IS_A1_A2_B2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0789177, 0.4454548, 0.0644008, 0.5302621, -0.4513444, 0.3810540
1: 0.0311321, 0.5851125, 0.0192177, 0.7222201, -0.6910880, 0.5658948
2: -0.0367235, 0.5172547, -0.0694884, 0.5835138, -0.6202373, 0.5867431
3: -0.1111584, 0.5267107, -0.1317451, 0.6470909, -0.7582493, 0.6584558
4: -0.1149590, 0.6824017, -0.1483382, 0.8069319, -0.9218909, 0.8307399

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_A2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_B2_B1_B1_B1

### Relational analysis result of IS_A1_A2_B2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5413930, upper bound: 0.4691153
time: 0.35 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_A2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_A2_B2_B1_B1_B1

### Relational analysis result of IS_A1_A2_B2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5602595, upper bound: 0.4677552
time: 0.36 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_A2_B2_B1_B1_B1

### Relational analysis result of IS_A1_A2_B2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5646571, upper bound: 0.5230881
time: 0.37 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_A2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_A2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_A2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 5

Time for candidate selection: 5.10 seconds

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B2_B1_B1_A1

### Relational analysis result of IS_A1_A2_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5707020, upper bound: 0.4652806
time: 0.38 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_A2

### Relational analysis result of IS_A1_A2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5706720, upper bound: 0.5448622
time: 0.38 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0789177, 0.4454548, -0.0054744, 0.7100763, -0.6311586, 0.4509293
1: 0.0311321, 0.5851125, -0.0693125, 0.9573513, -0.9262192, 0.6544250
2: -0.0367235, 0.5172547, -0.1748363, 0.7621485, -0.7988720, 0.6920910
3: -0.1111584, 0.5267107, -0.2408400, 0.9047788, -1.0159371, 0.7675507
4: -0.1149590, 0.6824017, -0.2802821, 1.0476310, -1.1625900, 0.9626838

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_A2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_B2_B1_B2_B1

### Relational analysis result of IS_A1_A2_B2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5413930, upper bound: 0.4691153
time: 0.36 seconds

## Relational analysis of IS_A1_A2_B2_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_A2_B2_B1_B2_B1

### Relational analysis result of IS_A1_A2_B2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5646571, upper bound: 0.5230881
time: 0.34 seconds

## Relational analysis of IS_A1_A2_B2_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_A2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_A2_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_A2_B2_B1_B2_B1

### Relational analysis result of IS_A1_A2_B2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5602595, upper bound: 0.4677552
time: 0.40 seconds

## Relational analysis of IS_A1_A2_B2_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_A2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_A2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 39

Time for candidate selection: 5.15 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B2_B1_B2_A1

### Relational analysis result of IS_A1_A2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5679063, upper bound: 0.5397815
time: 0.37 seconds

## Relational analysis of IS_A1_A2_B2_B1_B2_A2

### Relational analysis result of IS_A1_A2_B2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5598062, upper bound: 0.5425539
time: 0.35 seconds

## BFS IS instance: IS_A1_A2_B2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0789177, 0.4454548, 0.0793607, 0.4436836, -0.3647659, 0.3660941
1: 0.0311321, 0.5851125, 0.0318375, 0.5821407, -0.5510086, 0.5532750
2: -0.0367235, 0.5172547, -0.0354755, 0.5160056, -0.5527291, 0.5527302
3: -0.1111584, 0.5267107, -0.1102905, 0.5241475, -0.6353058, 0.6370012
4: -0.1149590, 0.6824017, -0.1139328, 0.6799669, -0.7949259, 0.7963345

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_A2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_A2_B2_B2_B1_A1

### Relational analysis result of IS_A1_A2_B2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5524581, upper bound: 0.5403838
time: 0.39 seconds

## Relational analysis of IS_A1_A2_B2_B2_B1_A2

### Relational analysis result of IS_A1_A2_B2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5539849, upper bound: 0.5402002
time: 0.36 seconds

## BFS IS instance: IS_A1_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0789177, 0.4454548, 0.0100145, 0.5674599, -0.4885422, 0.4354403
1: 0.0311321, 0.5851125, -0.0486783, 0.7368349, -0.7057028, 0.6337908
2: -0.0367235, 0.5172547, -0.1268167, 0.6409237, -0.6776472, 0.6440715
3: -0.1111584, 0.5267107, -0.2082636, 0.7057808, -0.8169392, 0.7349743
4: -0.1149590, 0.6824017, -0.2279943, 0.8359103, -0.9508693, 0.9103960

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_A2_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_A2_B2_B2_B2_B1

### Relational analysis result of IS_A1_A2_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5659468, upper bound: 0.5387858
time: 0.38 seconds

## Relational analysis of IS_A1_A2_B2_B2_B2_B2

### Relational analysis result of IS_A1_A2_B2_B2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5539849, upper bound: 0.5402002
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0069308, 0.7141405, 0.0873206, 0.4219245, -0.4288553, 0.6268200
1: -0.0711197, 0.9631418, 0.0395762, 0.5401090, -0.6112286, 0.9235656
2: -0.1773483, 0.7663486, -0.0181572, 0.5037278, -0.6810760, 0.7845058
3: -0.2441539, 0.9109904, -0.0989394, 0.4913468, -0.7355006, 1.0099298
4: -0.2841340, 1.0538890, -0.0983677, 0.6463114, -0.9304454, 1.1522567

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_B2_A1_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5111559, upper bound: 0.5684190
time: 0.36 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5272332, upper bound: 0.5598047
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0069308, 0.7141405, 0.0835698, 0.3964890, -0.4034199, 0.6305708
1: -0.0711197, 0.9631418, 0.0367923, 0.4783883, -0.5495080, 0.9263495
2: -0.1773483, 0.7663486, -0.0002279, 0.4968450, -0.6741933, 0.7665765
3: -0.2441539, 0.9109904, -0.0865010, 0.4496876, -0.6938415, 0.9974914
4: -0.2841340, 1.0538890, -0.0769091, 0.5933239, -0.8774580, 1.1307981

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 5

## Relational analysis of IS_A2_B1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_B2_A1_B2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5165278, upper bound: 0.5646490
time: 0.36 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2_B2

### Relational analysis result of IS_A2_B1_B2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5300057, upper bound: 0.5526206
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0083361, 0.5719668, 0.0873206, 0.4219245, -0.4135884, 0.4846462
1: -0.0506135, 0.7433118, 0.0395762, 0.5401090, -0.5907225, 0.7037356
2: -0.1297052, 0.6455543, -0.0181572, 0.5037278, -0.6334330, 0.6637115
3: -0.2118729, 0.7126579, -0.0989394, 0.4913468, -0.7032197, 0.8115973
4: -0.2322460, 0.8428500, -0.0983677, 0.6463114, -0.8785574, 0.9412177

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5415877, upper bound: 0.5654935
time: 0.42 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5304913, upper bound: 0.5754925
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0083361, 0.5719668, 0.0835698, 0.3964890, -0.3881530, 0.4883970
1: -0.0506135, 0.7433118, 0.0367923, 0.4783883, -0.5290018, 0.7065195
2: -0.1297052, 0.6455543, -0.0002279, 0.4968450, -0.6265502, 0.6457822
3: -0.2118729, 0.7126579, -0.0865010, 0.4496876, -0.6615605, 0.7991589
4: -0.2322460, 0.8428500, -0.0769091, 0.5933239, -0.8255700, 0.9197590

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 5

## Relational analysis of IS_A2_B1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_B2_A2_B2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5197859, upper bound: 0.5724233
time: 0.41 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5332638, upper bound: 0.5611240
time: 0.40 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 2.99 seconds
IS_A1_A2_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.99
Output dim: 0, lower bound: -0.5707020, upper bound: 0.4652806
IS_A1_A2_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.99
Output dim: 0, lower bound: -0.5706720, upper bound: 0.5448622
IS_A1_A2_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.99
Output dim: 0, lower bound: -0.5679063, upper bound: 0.5397815
IS_A1_A2_B2_B1_B2_A2, status: Status.VERIFIED, split count: 6, time: 2.99
Output dim: 0, lower bound: -0.5598062, upper bound: 0.5425539
IS_A1_A2_B2_B2_B1_A1, status: Status.VERIFIED, split count: 6, time: 2.99
Output dim: 0, lower bound: -0.5524581, upper bound: 0.5403838
IS_A1_A2_B2_B2_B1_A2, status: Status.VERIFIED, split count: 6, time: 2.99
Output dim: 0, lower bound: -0.5539849, upper bound: 0.5402002
IS_A1_A2_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.99
Output dim: 0, lower bound: -0.5659468, upper bound: 0.5387858
IS_A1_A2_B2_B2_B2_B2, status: Status.VERIFIED, split count: 6, time: 2.99
Output dim: 0, lower bound: -0.5539849, upper bound: 0.5402002
IS_A2_B1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.99
Output dim: 0, lower bound: -0.5111559, upper bound: 0.5684190
IS_A2_B1_B2_A1_B1_B2, status: Status.VERIFIED, split count: 6, time: 2.99
Output dim: 0, lower bound: -0.5272332, upper bound: 0.5598047
IS_A2_B1_B2_A1_B2_B1, status: Status.VERIFIED, split count: 6, time: 2.99
Output dim: 0, lower bound: -0.5165278, upper bound: 0.5646490
IS_A2_B1_B2_A1_B2_B2, status: Status.VERIFIED, split count: 6, time: 2.99
Output dim: 0, lower bound: -0.5300057, upper bound: 0.5526206
IS_A2_B1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.99
Output dim: 0, lower bound: -0.5415877, upper bound: 0.5654935
IS_A2_B1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.99
Output dim: 0, lower bound: -0.5304913, upper bound: 0.5754925
IS_A2_B1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.99
Output dim: 0, lower bound: -0.5197859, upper bound: 0.5724233
IS_A2_B1_B2_A2_B2_B2, status: Status.VERIFIED, split count: 6, time: 2.99
Output dim: 0, lower bound: -0.5332638, upper bound: 0.5611240

## BFS IS instance: IS_A1_A2_B2_B1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0881433, 0.3505849, 0.0646830, 0.5288626, -0.4407192, 0.2859019
1: 0.0561979, 0.3945956, 0.0195227, 0.7196292, -0.6634313, 0.3750729
2: 0.0497189, 0.4406298, -0.0684017, 0.5826666, -0.5329477, 0.5090315
3: -0.0570518, 0.4040877, -0.1308922, 0.6449826, -0.7020344, 0.5349798
4: -0.0166695, 0.4652953, -0.1473095, 0.8047376, -0.8214071, 0.6126049

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B2_B1_B1_A1_A1

### Relational analysis result of IS_A1_A2_B2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5677527, upper bound: 0.4893037
time: 0.41 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B2_B1_B1_A1_A1

### Relational analysis result of IS_A1_A2_B2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5696524, upper bound: 0.4702711
time: 0.37 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_A1_A2

### Relational analysis result of IS_A1_A2_B2_B1_B1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5617467, upper bound: 0.4863662
time: 0.39 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0797181, 0.4446977, 0.0644008, 0.5302621, -0.4505440, 0.3802969
1: 0.0318470, 0.5838390, 0.0192177, 0.7222201, -0.6903731, 0.5646213
2: -0.0357902, 0.5167565, -0.0694884, 0.5835138, -0.6193040, 0.5862449
3: -0.1101345, 0.5255548, -0.1317451, 0.6470909, -0.7572254, 0.6572999
4: -0.1140501, 0.6814760, -0.1483382, 0.8069319, -0.9209820, 0.8298142

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B2_B1_B1_A2_A1

### Relational analysis result of IS_A1_A2_B2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5674961, upper bound: 0.5696298
time: 0.38 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_A2_A2

### Relational analysis result of IS_A1_A2_B2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5595722, upper bound: 0.5703417
time: 0.35 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0873206, 0.4219245, -0.0054744, 0.7100763, -0.6227558, 0.4273989
1: 0.0395762, 0.5401090, -0.0693125, 0.9573513, -0.9177752, 0.6094214
2: -0.0181572, 0.5037278, -0.1748363, 0.7621485, -0.7803057, 0.6785641
3: -0.0989394, 0.4913468, -0.2408400, 0.9047788, -1.0037181, 0.7321868
4: -0.0983677, 0.6463114, -0.2802821, 1.0476310, -1.1459987, 0.9265935

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B2_B1_B2_A1_A1

### Relational analysis result of IS_A1_A2_B2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5683143, upper bound: 0.5109562
time: 0.39 seconds

## Relational analysis of IS_A1_A2_B2_B1_B2_A1_A2

### Relational analysis result of IS_A1_A2_B2_B1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5596943, upper bound: 0.5270336
time: 0.44 seconds

## BFS IS instance: IS_A1_A2_B2_B2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0789177, 0.4454548, 0.0294979, 0.5303939, -0.4514762, 0.4159570
1: 0.0311321, 0.5851125, -0.0272393, 0.6770903, -0.6459582, 0.6123518
2: -0.0367235, 0.5172547, -0.0920497, 0.6105885, -0.6473120, 0.6093044
3: -0.1111584, 0.5267107, -0.1739870, 0.6456757, -0.7568341, 0.7006977
4: -0.1149590, 0.6824017, -0.1849493, 0.7805903, -0.8955493, 0.8673509

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_B2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_A2_B2_B2_B2_B1_A1

### Relational analysis result of IS_A1_A2_B2_B2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5622473, upper bound: 0.5387858
time: 0.40 seconds

## Relational analysis of IS_A1_A2_B2_B2_B2_B1_A2

### Relational analysis result of IS_A1_A2_B2_B2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5622475, upper bound: 0.5387858
time: 0.47 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0069308, 0.7141405, 0.0965443, 0.3969810, -0.4039118, 0.6175963
1: -0.0711197, 0.9631418, 0.0565891, 0.4885263, -0.5596460, 0.9065527
2: -0.1773483, 0.7663486, 0.0083284, 0.4923123, -0.6696606, 0.7580202
3: -0.2441539, 0.9109904, -0.0780025, 0.4490798, -0.6932336, 0.9889928
4: -0.2841340, 1.0538890, -0.0766807, 0.5934064, -0.8775404, 1.1305697

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_B2_A1_B1_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5111559, upper bound: 0.5684190
time: 0.35 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_B1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5111559, upper bound: 0.5646490
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0433788, 0.4866467, 0.0873206, 0.4219245, -0.3785456, 0.3993261
1: -0.0069845, 0.5967795, 0.0395762, 0.5401090, -0.5470935, 0.5572033
2: -0.0581915, 0.5744838, -0.0181572, 0.5037278, -0.5619193, 0.5926410
3: -0.1468202, 0.5730036, -0.0989394, 0.4913468, -0.6381670, 0.6719431
4: -0.1378696, 0.6996939, -0.0983677, 0.6463114, -0.7841810, 0.7980616

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5415877, upper bound: 0.5654935
time: 0.43 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5415877, upper bound: 0.5507922
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0233569, 0.5570214, 0.0873206, 0.4219245, -0.3985676, 0.4697009
1: -0.0314330, 0.7206672, 0.0395762, 0.5401090, -0.5715420, 0.6810911
2: -0.1071281, 0.6316106, -0.0181572, 0.5037278, -0.6108559, 0.6497678
3: -0.1827013, 0.6866775, -0.0989394, 0.4913468, -0.6740481, 0.7856169
4: -0.2049841, 0.8204701, -0.0983677, 0.6463114, -0.8512955, 0.9188378

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5304913, upper bound: 0.5754925
time: 0.42 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5247494, upper bound: 0.5611240
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0083361, 0.5719668, 0.0921624, 0.3807045, -0.3723685, 0.4798044
1: -0.0506135, 0.7433118, 0.0501726, 0.4466347, -0.4972482, 0.6931393
2: -0.1297052, 0.6455543, 0.0185485, 0.4862787, -0.6159838, 0.6270058
3: -0.2118729, 0.7126579, -0.0701120, 0.4227494, -0.6346223, 0.7827699
4: -0.2322460, 0.8428500, -0.0608306, 0.5514524, -0.7836984, 0.9036806

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B1

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5724233
time: 0.39 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5197859, upper bound: 0.5724233
time: 0.41 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 2.51 seconds
IS_A1_A2_B2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 2.51
Output dim: 0, lower bound: -0.5696524, upper bound: 0.4702711
IS_A1_A2_B2_B1_B1_A1_A2, status: Status.VERIFIED, split count: 7, time: 2.51
Output dim: 0, lower bound: -0.5617467, upper bound: 0.4863662
IS_A1_A2_B2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 2.51
Output dim: 0, lower bound: -0.5674961, upper bound: 0.5696298
IS_A1_A2_B2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.51
Output dim: 0, lower bound: -0.5595722, upper bound: 0.5703417
IS_A1_A2_B2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 2.51
Output dim: 0, lower bound: -0.5683143, upper bound: 0.5109562
IS_A1_A2_B2_B1_B2_A1_A2, status: Status.VERIFIED, split count: 7, time: 2.51
Output dim: 0, lower bound: -0.5596943, upper bound: 0.5270336
IS_A1_A2_B2_B2_B2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.51
Output dim: 0, lower bound: -0.5622473, upper bound: 0.5387858
IS_A1_A2_B2_B2_B2_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.51
Output dim: 0, lower bound: -0.5622475, upper bound: 0.5387858
IS_A2_B1_B2_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.51
Output dim: 0, lower bound: -0.5111559, upper bound: 0.5684190
IS_A2_B1_B2_A1_B1_B1_B2, status: Status.VERIFIED, split count: 7, time: 2.51
Output dim: 0, lower bound: -0.5111559, upper bound: 0.5646490
IS_A2_B1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.51
Output dim: 0, lower bound: -0.5415877, upper bound: 0.5654935
IS_A2_B1_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 7, time: 2.51
Output dim: 0, lower bound: -0.5415877, upper bound: 0.5507922
IS_A2_B1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.51
Output dim: 0, lower bound: -0.5304913, upper bound: 0.5754925
IS_A2_B1_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 7, time: 2.51
Output dim: 0, lower bound: -0.5247494, upper bound: 0.5611240
IS_A2_B1_B2_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.51
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5724233
IS_A2_B1_B2_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.51
Output dim: 0, lower bound: -0.5197859, upper bound: 0.5724233

## BFS IS instance: IS_A1_A2_B2_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0945764, 0.3505142, 0.0644008, 0.5302621, -0.4356858, 0.2848256
1: 0.0685086, 0.3967934, 0.0192177, 0.7222201, -0.6537115, 0.3775756
2: 0.0585651, 0.4334365, -0.0694884, 0.5835138, -0.5249487, 0.5029249
3: -0.0406969, 0.4090275, -0.1317451, 0.6470909, -0.6877878, 0.5407726
4: -0.0057726, 0.4402502, -0.1483382, 0.8069319, -0.8127044, 0.5885884

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B2_B1_B1_A1_A1_A1

### Relational analysis result of IS_A1_A2_B2_B1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5696524, upper bound: 0.4702711
time: 0.38 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_A1_A1_A2

### Relational analysis result of IS_A1_A2_B2_B1_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5694090, upper bound: 0.4702711
time: 0.36 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: 0.0880811, 0.4212151, 0.0644008, 0.5302621, -0.4421810, 0.3568143
1: 0.0402591, 0.5388805, 0.0192177, 0.7222201, -0.6819610, 0.5196627
2: -0.0172656, 0.5033177, -0.0694884, 0.5835138, -0.6007794, 0.5728061
3: -0.0979600, 0.4902630, -0.1317451, 0.6470909, -0.7450509, 0.6220081
4: -0.0975381, 0.6454344, -0.1483382, 0.8069319, -0.9044700, 0.7937726

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B2_B1_B1_A2_A1_A1

### Relational analysis result of IS_A1_A2_B2_B1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5674961, upper bound: 0.4893037
time: 0.39 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_A2_A1_A2

### Relational analysis result of IS_A1_A2_B2_B1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5674961, upper bound: 0.5696298
time: 0.38 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: 0.0842948, 0.3956742, 0.0644008, 0.5302621, -0.4459673, 0.3312734
1: 0.0380552, 0.4769628, 0.0192177, 0.7222201, -0.6841649, 0.4577451
2: 0.0011241, 0.4962993, -0.0694884, 0.5835138, -0.5823897, 0.5657877
3: -0.0846072, 0.4483558, -0.1317451, 0.6470909, -0.7316982, 0.5801009
4: -0.0755858, 0.5922823, -0.1483382, 0.8069319, -0.8825177, 0.7406205

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B2_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B2_B1_B1_A2_A2_A1

### Relational analysis result of IS_A1_A2_B2_B1_B1_A2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5586579, upper bound: 0.5460839
time: 0.47 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_A2_A2_A2

### Relational analysis result of IS_A1_A2_B2_B1_B1_A2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5464140, upper bound: 0.5589715
time: 0.39 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B2_A1_A1

### Backsubstitution after applying IS history:
0: 0.0965443, 0.3969810, -0.0054744, 0.7100763, -0.6135321, 0.4024554
1: 0.0565891, 0.4885263, -0.0693125, 0.9573513, -0.9007622, 0.5578388
2: 0.0083284, 0.4923123, -0.1748363, 0.7621485, -0.7538201, 0.6671486
3: -0.0780025, 0.4490798, -0.2408400, 0.9047788, -0.9827812, 0.6899198
4: -0.0766807, 0.5934064, -0.2802821, 1.0476310, -1.1243117, 0.8736885

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B2_B1_B2_A1_A1_A1

### Relational analysis result of IS_A1_A2_B2_B1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5683143, upper bound: 0.5109562
time: 0.40 seconds

## Relational analysis of IS_A1_A2_B2_B1_B2_A1_A1_A2

### Relational analysis result of IS_A1_A2_B2_B1_B2_A1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5645366, upper bound: 0.5109562
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0069308, 0.7141405, 0.0965443, 0.3969810, -0.4039118, 0.6175963
1: -0.0711197, 0.9631418, 0.0565891, 0.4885263, -0.5596460, 0.9065527
2: -0.1773483, 0.7663486, 0.0083284, 0.4923123, -0.6696606, 0.7580202
3: -0.2441539, 0.9109904, -0.0780025, 0.4490798, -0.6932336, 0.9889928
4: -0.2841340, 1.0538890, -0.0766807, 0.5934064, -0.8775404, 1.1305697

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_B2_A1_B1_B1_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5111559, upper bound: 0.5684190
time: 0.36 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_B1_B1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_B1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5111559, upper bound: 0.5598047
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0433788, 0.4866467, 0.0873206, 0.4219245, -0.3785456, 0.3993261
1: -0.0069845, 0.5967795, 0.0395762, 0.5401090, -0.5470935, 0.5572033
2: -0.0581915, 0.5744838, -0.0181572, 0.5037278, -0.5619193, 0.5926410
3: -0.1468202, 0.5730036, -0.0989394, 0.4913468, -0.6381670, 0.6719431
4: -0.1378696, 0.6996939, -0.0983677, 0.6463114, -0.7841810, 0.7980616

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5415877, upper bound: 0.5654935
time: 0.43 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5304913, upper bound: 0.5654935
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0233569, 0.5570214, 0.0873206, 0.4219245, -0.3985676, 0.4697009
1: -0.0314330, 0.7206672, 0.0395762, 0.5401090, -0.5715420, 0.6810911
2: -0.1071281, 0.6316106, -0.0181572, 0.5037278, -0.6108559, 0.6497678
3: -0.1827013, 0.6866775, -0.0989394, 0.4913468, -0.6740481, 0.7856169
4: -0.2049841, 0.8204701, -0.0983677, 0.6463114, -0.8512955, 0.9188378

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5304913, upper bound: 0.5654935
time: 0.39 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5304913, upper bound: 0.5754925
time: 0.48 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0083361, 0.5719668, 0.0969255, 0.3960601, -0.3877240, 0.4750413
1: -0.0506135, 0.7433118, 0.0570602, 0.4867831, -0.5373967, 0.6862516
2: -0.1297052, 0.6455543, 0.0090892, 0.4917682, -0.6214734, 0.6364651
3: -0.2118729, 0.7126579, -0.0774024, 0.4474990, -0.6593719, 0.7900603
4: -0.2322460, 0.8428500, -0.0759368, 0.5922844, -0.8245305, 0.9187868

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B1_B1

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5797783
time: 0.41 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B1_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5754925
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0083361, 0.5719668, 0.0921624, 0.3807045, -0.3723685, 0.4798044
1: -0.0506135, 0.7433118, 0.0501726, 0.4466347, -0.4972482, 0.6931393
2: -0.1297052, 0.6455543, 0.0185485, 0.4862787, -0.6159838, 0.6270058
3: -0.2118729, 0.7126579, -0.0701120, 0.4227494, -0.6346223, 0.7827699
4: -0.2322460, 0.8428500, -0.0608306, 0.5514524, -0.7836984, 0.9036806

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5197859, upper bound: 0.5724233
time: 0.41 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5197859, upper bound: 0.5611240
time: 0.37 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 2.59 seconds
IS_A1_A2_B2_B1_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.5696524, upper bound: 0.4702711
IS_A1_A2_B2_B1_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.5694090, upper bound: 0.4702711
IS_A1_A2_B2_B1_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.5674961, upper bound: 0.4893037
IS_A1_A2_B2_B1_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.5674961, upper bound: 0.5696298
IS_A1_A2_B2_B1_B1_A2_A2_A1, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.5586579, upper bound: 0.5460839
IS_A1_A2_B2_B1_B1_A2_A2_A2, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.5464140, upper bound: 0.5589715
IS_A1_A2_B2_B1_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.5683143, upper bound: 0.5109562
IS_A1_A2_B2_B1_B2_A1_A1_A2, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.5645366, upper bound: 0.5109562
IS_A2_B1_B2_A1_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.5111559, upper bound: 0.5684190
IS_A2_B1_B2_A1_B1_B1_B1_B2, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.5111559, upper bound: 0.5598047
IS_A2_B1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.5415877, upper bound: 0.5654935
IS_A2_B1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.5304913, upper bound: 0.5654935
IS_A2_B1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.5304913, upper bound: 0.5654935
IS_A2_B1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.5304913, upper bound: 0.5754925
IS_A2_B1_B2_A2_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5797783
IS_A2_B1_B2_A2_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5754925
IS_A2_B1_B2_A2_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.5197859, upper bound: 0.5724233
IS_A2_B1_B2_A2_B2_B1_B2_B2, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.5197859, upper bound: 0.5611240

## BFS IS instance: IS_A1_A2_B2_B1_B1_A1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0945764, 0.3505142, 0.0646830, 0.5288626, -0.4342862, 0.2845256
1: 0.0685086, 0.3967934, 0.0195227, 0.7196292, -0.6511206, 0.3772707
2: 0.0585651, 0.4334365, -0.0684017, 0.5826666, -0.5241014, 0.5018382
3: -0.0406969, 0.4090275, -0.1308922, 0.6449826, -0.6856795, 0.5399196
4: -0.0057726, 0.4402502, -0.1473095, 0.8047376, -0.8105102, 0.5875597

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B2_B1_B1_A1_A1_A1_A1

### Relational analysis result of IS_A1_A2_B2_B1_B1_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5663972, upper bound: 0.4605393
time: 0.40 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_A1_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B2_B1_B1_A1_A1_A1_A1

### Relational analysis result of IS_A1_A2_B2_B1_B1_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5696524, upper bound: 0.4702711
time: 0.39 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_A1_A1_A1_A2

### Relational analysis result of IS_A1_A2_B2_B1_B1_A1_A1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5617467, upper bound: 0.4702711
time: 0.37 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B1_A1_A1_A2

### Backsubstitution after applying IS history:
0: 0.0924387, 0.3566364, 0.0644008, 0.5302621, -0.4378234, 0.2922356
1: 0.0503964, 0.4023641, 0.0192177, 0.7222201, -0.6718237, 0.3831464
2: 0.0225284, 0.5009353, -0.0694884, 0.5835138, -0.5609854, 0.5704237
3: -0.0864439, 0.3917682, -0.1317451, 0.6470909, -0.7335348, 0.5235133
4: -0.0891201, 0.5688827, -0.1483382, 0.8069319, -0.8960520, 0.7172209

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B2_B1_B1_A1_A1_A2_A1

### Relational analysis result of IS_A1_A2_B2_B1_B1_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5694090, upper bound: 0.4702711
time: 0.38 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_A1_A1_A2_A2

### Relational analysis result of IS_A1_A2_B2_B1_B1_A1_A1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5613755, upper bound: 0.4702711
time: 0.38 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B1_A2_A1_A1

### Backsubstitution after applying IS history:
0: 0.0935934, 0.3446476, 0.0646830, 0.5288626, -0.4352692, 0.2799646
1: 0.0695772, 0.3865252, 0.0195227, 0.7196292, -0.6500520, 0.3670025
2: 0.0587194, 0.4287091, -0.0684017, 0.5826666, -0.5239472, 0.4971108
3: -0.0373166, 0.3964920, -0.1308922, 0.6449826, -0.6822992, 0.5273842
4: -0.0043771, 0.4454577, -0.1473095, 0.8047376, -0.8091147, 0.5927672

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B2_B1_B1_A2_A1_A1_A1

### Relational analysis result of IS_A1_A2_B2_B1_B1_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5677527, upper bound: 0.4893037
time: 0.43 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_A2_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B2_B1_B1_A2_A1_A1_A1

### Relational analysis result of IS_A1_A2_B2_B1_B1_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5663972, upper bound: 0.4605393
time: 0.39 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_A2_A1_A1_A2

### Relational analysis result of IS_A1_A2_B2_B1_B1_A2_A1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5583578, upper bound: 0.4779336
time: 0.42 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B1_A2_A1_A2

### Backsubstitution after applying IS history:
0: 0.0880811, 0.4212151, 0.0644008, 0.5302621, -0.4421810, 0.3568143
1: 0.0402591, 0.5388805, 0.0192177, 0.7222201, -0.6819610, 0.5196627
2: -0.0172656, 0.5033177, -0.0694884, 0.5835138, -0.6007794, 0.5728061
3: -0.0979600, 0.4902630, -0.1317451, 0.6470909, -0.7450509, 0.6220081
4: -0.0975381, 0.6454344, -0.1483382, 0.8069319, -0.9044700, 0.7937726

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B2_B1_B1_A2_A1_A2_A1

### Relational analysis result of IS_A1_A2_B2_B1_B1_A2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5674961, upper bound: 0.5696298
time: 0.40 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_A2_A1_A2_A2

### Relational analysis result of IS_A1_A2_B2_B1_B1_A2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5595722, upper bound: 0.5696298
time: 0.41 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B2_A1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0965443, 0.3969810, -0.0054744, 0.7100763, -0.6135321, 0.4024554
1: 0.0565891, 0.4885263, -0.0693125, 0.9573513, -0.9007622, 0.5578388
2: 0.0083284, 0.4923123, -0.1748363, 0.7621485, -0.7538201, 0.6671486
3: -0.0780025, 0.4490798, -0.2408400, 0.9047788, -0.9827812, 0.6899198
4: -0.0766807, 0.5934064, -0.2802821, 1.0476310, -1.1243117, 0.8736885

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B2_B1_B2_A1_A1_A1_A1

### Relational analysis result of IS_A1_A2_B2_B1_B2_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5683143, upper bound: 0.5109562
time: 0.41 seconds

## Relational analysis of IS_A1_A2_B2_B1_B2_A1_A1_A1_A2

### Relational analysis result of IS_A1_A2_B2_B1_B2_A1_A1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5596943, upper bound: 0.5109562
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0069308, 0.7141405, 0.0965443, 0.3969810, -0.4039118, 0.6175963
1: -0.0711197, 0.9631418, 0.0565891, 0.4885263, -0.5596460, 0.9065527
2: -0.1773483, 0.7663486, 0.0083284, 0.4923123, -0.6696606, 0.7580202
3: -0.2441539, 0.9109904, -0.0780025, 0.4490798, -0.6932336, 0.9889928
4: -0.2841340, 1.0538890, -0.0766807, 0.5934064, -0.8775404, 1.1305697

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_B2_A1_B1_B1_B1_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5111559, upper bound: 0.5684190
time: 0.37 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_B1_B1_B1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_B1_B1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5111559, upper bound: 0.5646490
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0433788, 0.4866467, 0.0873206, 0.4219245, -0.3785456, 0.3993261
1: -0.0069845, 0.5967795, 0.0395762, 0.5401090, -0.5470935, 0.5572033
2: -0.0581915, 0.5744838, -0.0181572, 0.5037278, -0.5619193, 0.5926410
3: -0.1468202, 0.5730036, -0.0989394, 0.4913468, -0.6381670, 0.6719431
4: -0.1378696, 0.6996939, -0.0983677, 0.6463114, -0.7841810, 0.7980616

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_B2_A2_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5415877, upper bound: 0.5654935
time: 0.41 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5415877, upper bound: 0.5507922
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0283030, 0.5522523, 0.0873206, 0.4219245, -0.3936214, 0.4649317
1: -0.0250288, 0.7141986, 0.0395762, 0.5401090, -0.5651378, 0.6746224
2: -0.0988264, 0.6263493, -0.0181572, 0.5037278, -0.6025542, 0.6445066
3: -0.1721501, 0.6781709, -0.0989394, 0.4913468, -0.6634969, 0.7771103
4: -0.1945045, 0.8138232, -0.0983677, 0.6463114, -0.8408159, 0.9121909

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5304913, upper bound: 0.5654935
time: 0.39 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5304913, upper bound: 0.5507922
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0433788, 0.4866467, 0.0873206, 0.4219245, -0.3785456, 0.3993261
1: -0.0069845, 0.5967795, 0.0395762, 0.5401090, -0.5470935, 0.5572033
2: -0.0581915, 0.5744838, -0.0181572, 0.5037278, -0.5619193, 0.5926410
3: -0.1468202, 0.5730036, -0.0989394, 0.4913468, -0.6381670, 0.6719431
4: -0.1378696, 0.6996939, -0.0983677, 0.6463114, -0.7841810, 0.7980616

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_B2_A2_B1_A2_B1_A1_A1

### Relational analysis result of IS_A2_B1_B2_A2_B1_A2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5415877, upper bound: 0.5653172
time: 0.40 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_A2_B1_A1_A2

### Relational analysis result of IS_A2_B1_B2_A2_B1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5415877, upper bound: 0.5654935
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0233569, 0.5570214, 0.0873206, 0.4219245, -0.3985676, 0.4697009
1: -0.0314330, 0.7206672, 0.0395762, 0.5401090, -0.5715420, 0.6810911
2: -0.1071281, 0.6316106, -0.0181572, 0.5037278, -0.6108559, 0.6497678
3: -0.1827013, 0.6866775, -0.0989394, 0.4913468, -0.6740481, 0.7856169
4: -0.2049841, 0.8204701, -0.0983677, 0.6463114, -0.8512955, 0.9188378

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5304913, upper bound: 0.5754925
time: 0.42 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5304913, upper bound: 0.5611240
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B2_B1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0083361, 0.5719668, 0.0969255, 0.3960601, -0.3877240, 0.4750413
1: -0.0506135, 0.7433118, 0.0570602, 0.4867831, -0.5373967, 0.6862516
2: -0.1297052, 0.6455543, 0.0090892, 0.4917682, -0.6214734, 0.6364651
3: -0.2118729, 0.7126579, -0.0774024, 0.4474990, -0.6593719, 0.7900603
4: -0.2322460, 0.8428500, -0.0759368, 0.5922844, -0.8245305, 0.9187868

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B1_B1_B1

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5797783
time: 0.46 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B1_B1_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5724233
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B2_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0083361, 0.5719668, 0.0937147, 0.4119777, -0.4036416, 0.4782521
1: -0.0506135, 0.7433118, 0.0489067, 0.5234078, -0.5740213, 0.6944051
2: -0.1297052, 0.6455543, -0.0067511, 0.4970919, -0.6267971, 0.6523054
3: -0.2118729, 0.7126579, -0.0869917, 0.4748205, -0.6866934, 0.7996497
4: -0.2322460, 0.8428500, -0.0889151, 0.6335514, -0.8657974, 0.9317651

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5754925
time: 0.40 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5754925
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B2_B1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0083361, 0.5719668, 0.0921624, 0.3807045, -0.3723685, 0.4798044
1: -0.0506135, 0.7433118, 0.0501726, 0.4466347, -0.4972482, 0.6931393
2: -0.1297052, 0.6455543, 0.0185485, 0.4862787, -0.6159838, 0.6270058
3: -0.2118729, 0.7126579, -0.0701120, 0.4227494, -0.6346223, 0.7827699
4: -0.2322460, 0.8428500, -0.0608306, 0.5514524, -0.7836984, 0.9036806

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B2_B1_B1

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5724233
time: 0.41 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B2_B1_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5197859, upper bound: 0.5724233
time: 0.45 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 2.68 seconds
IS_A1_A2_B2_B1_B1_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 9, time: 2.68
Output dim: 0, lower bound: -0.5696524, upper bound: 0.4702711
IS_A1_A2_B2_B1_B1_A1_A1_A1_A2, status: Status.VERIFIED, split count: 9, time: 2.68
Output dim: 0, lower bound: -0.5617467, upper bound: 0.4702711
IS_A1_A2_B2_B1_B1_A1_A1_A2_A1, status: Status.UNKNOWN, split count: 9, time: 2.68
Output dim: 0, lower bound: -0.5694090, upper bound: 0.4702711
IS_A1_A2_B2_B1_B1_A1_A1_A2_A2, status: Status.VERIFIED, split count: 9, time: 2.68
Output dim: 0, lower bound: -0.5613755, upper bound: 0.4702711
IS_A1_A2_B2_B1_B1_A2_A1_A1_A1, status: Status.UNKNOWN, split count: 9, time: 2.68
Output dim: 0, lower bound: -0.5663972, upper bound: 0.4605393
IS_A1_A2_B2_B1_B1_A2_A1_A1_A2, status: Status.VERIFIED, split count: 9, time: 2.68
Output dim: 0, lower bound: -0.5583578, upper bound: 0.4779336
IS_A1_A2_B2_B1_B1_A2_A1_A2_A1, status: Status.UNKNOWN, split count: 9, time: 2.68
Output dim: 0, lower bound: -0.5674961, upper bound: 0.5696298
IS_A1_A2_B2_B1_B1_A2_A1_A2_A2, status: Status.UNKNOWN, split count: 9, time: 2.68
Output dim: 0, lower bound: -0.5595722, upper bound: 0.5696298
IS_A1_A2_B2_B1_B2_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 9, time: 2.68
Output dim: 0, lower bound: -0.5683143, upper bound: 0.5109562
IS_A1_A2_B2_B1_B2_A1_A1_A1_A2, status: Status.VERIFIED, split count: 9, time: 2.68
Output dim: 0, lower bound: -0.5596943, upper bound: 0.5109562
IS_A2_B1_B2_A1_B1_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 9, time: 2.68
Output dim: 0, lower bound: -0.5111559, upper bound: 0.5684190
IS_A2_B1_B2_A1_B1_B1_B1_B1_B2, status: Status.VERIFIED, split count: 9, time: 2.68
Output dim: 0, lower bound: -0.5111559, upper bound: 0.5646490
IS_A2_B1_B2_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 9, time: 2.68
Output dim: 0, lower bound: -0.5415877, upper bound: 0.5654935
IS_A2_B1_B2_A2_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 9, time: 2.68
Output dim: 0, lower bound: -0.5415877, upper bound: 0.5507922
IS_A2_B1_B2_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 9, time: 2.68
Output dim: 0, lower bound: -0.5304913, upper bound: 0.5654935
IS_A2_B1_B2_A2_B1_A1_B1_A2_B2, status: Status.VERIFIED, split count: 9, time: 2.68
Output dim: 0, lower bound: -0.5304913, upper bound: 0.5507922
IS_A2_B1_B2_A2_B1_A2_B1_A1_A1, status: Status.VERIFIED, split count: 9, time: 2.68
Output dim: 0, lower bound: -0.5415877, upper bound: 0.5653172
IS_A2_B1_B2_A2_B1_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 9, time: 2.68
Output dim: 0, lower bound: -0.5415877, upper bound: 0.5654935
IS_A2_B1_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 9, time: 2.68
Output dim: 0, lower bound: -0.5304913, upper bound: 0.5754925
IS_A2_B1_B2_A2_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 9, time: 2.68
Output dim: 0, lower bound: -0.5304913, upper bound: 0.5611240
IS_A2_B1_B2_A2_B2_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 9, time: 2.68
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5797783
IS_A2_B1_B2_A2_B2_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 9, time: 2.68
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5724233
IS_A2_B1_B2_A2_B2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.68
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5754925
IS_A2_B1_B2_A2_B2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.68
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5754925
IS_A2_B1_B2_A2_B2_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 9, time: 2.68
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5724233
IS_A2_B1_B2_A2_B2_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 9, time: 2.68
Output dim: 0, lower bound: -0.5197859, upper bound: 0.5724233

## BFS IS instance: IS_A1_A2_B2_B1_B1_A1_A1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0945764, 0.3505142, 0.0644008, 0.5302621, -0.4356858, 0.2848256
1: 0.0685086, 0.3967934, 0.0192177, 0.7222201, -0.6537115, 0.3775756
2: 0.0585651, 0.4334365, -0.0694884, 0.5835138, -0.5249487, 0.5029249
3: -0.0406969, 0.4090275, -0.1317451, 0.6470909, -0.6877878, 0.5407726
4: -0.0057726, 0.4402502, -0.1483382, 0.8069319, -0.8127044, 0.5885884

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B2_B1_B1_A1_A1_A1_A1_A1

### Relational analysis result of IS_A1_A2_B2_B1_B1_A1_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5696524, upper bound: 0.4702711
time: 0.40 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_A1_A1_A1_A1_A2

### Relational analysis result of IS_A1_A2_B2_B1_B1_A1_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5694090, upper bound: 0.4702711
time: 0.39 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B1_A1_A1_A2_A1

### Backsubstitution after applying IS history:
0: 0.0924387, 0.3566364, 0.0644008, 0.5302621, -0.4378234, 0.2922356
1: 0.0503964, 0.4023641, 0.0192177, 0.7222201, -0.6718237, 0.3831464
2: 0.0225284, 0.5009353, -0.0694884, 0.5835138, -0.5609854, 0.5704237
3: -0.0864439, 0.3917682, -0.1317451, 0.6470909, -0.7335348, 0.5235133
4: -0.0891201, 0.5688827, -0.1483382, 0.8069319, -0.8960520, 0.7172209

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B2_B1_B1_A1_A1_A2_A1_A1

### Relational analysis result of IS_A1_A2_B2_B1_B1_A1_A1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5694090, upper bound: 0.4702711
time: 0.39 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_A1_A1_A2_A1_A2

### Relational analysis result of IS_A1_A2_B2_B1_B1_A1_A1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5694090, upper bound: 0.4702711
time: 0.38 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B1_A2_A1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0998964, 0.3467079, 0.0644008, 0.5302621, -0.4303657, 0.2823071
1: 0.0816488, 0.3923656, 0.0192177, 0.7222201, -0.6405713, 0.3731478
2: 0.0676711, 0.4236782, -0.0694884, 0.5835138, -0.5158427, 0.4931666
3: -0.0211008, 0.4035573, -0.1317451, 0.6470909, -0.6681917, 0.5353024
4: 0.0067785, 0.4217074, -0.1483382, 0.8069319, -0.8001534, 0.5700456

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B2_B1_B1_A2_A1_A1_A1_A1

### Relational analysis result of IS_A1_A2_B2_B1_B1_A2_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5663972, upper bound: 0.4605393
time: 0.41 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_A2_A1_A1_A1_A2

### Relational analysis result of IS_A1_A2_B2_B1_B1_A2_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5662008, upper bound: 0.4605393
time: 0.40 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B1_A2_A1_A2_A1

### Backsubstitution after applying IS history:
0: 0.0880811, 0.4212151, 0.0644008, 0.5302621, -0.4421810, 0.3568143
1: 0.0402591, 0.5388805, 0.0192177, 0.7222201, -0.6819610, 0.5196627
2: -0.0172656, 0.5033177, -0.0694884, 0.5835138, -0.6007794, 0.5728061
3: -0.0979600, 0.4902630, -0.1317451, 0.6470909, -0.7450509, 0.6220081
4: -0.0975381, 0.6454344, -0.1483382, 0.8069319, -0.9044700, 0.7937726

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B2_B1_B1_A2_A1_A2_A1_A1

### Relational analysis result of IS_A1_A2_B2_B1_B1_A2_A1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5674961, upper bound: 0.4893037
time: 0.46 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_A2_A1_A2_A1_A2

### Relational analysis result of IS_A1_A2_B2_B1_B1_A2_A1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5674961, upper bound: 0.5696298
time: 0.40 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B1_A2_A1_A2_A2

### Backsubstitution after applying IS history:
0: 0.0845213, 0.3948262, 0.0644008, 0.5302621, -0.4457408, 0.3304254
1: 0.0383507, 0.4757941, 0.0192177, 0.7222201, -0.6838694, 0.4565763
2: 0.0013437, 0.4952822, -0.0694884, 0.5835138, -0.5821701, 0.5647706
3: -0.0842046, 0.4470655, -0.1317451, 0.6470909, -0.7312955, 0.5788106
4: -0.0750232, 0.5913191, -0.1483382, 0.8069319, -0.8819550, 0.7396573

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B2_B1_B1_A2_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B2_B1_B1_A2_A1_A2_A2_A1

### Relational analysis result of IS_A1_A2_B2_B1_B1_A2_A1_A2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5586579, upper bound: 0.5421828
time: 0.39 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_A2_A1_A2_A2_A2

### Relational analysis result of IS_A1_A2_B2_B1_B1_A2_A1_A2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5464140, upper bound: 0.5582596
time: 0.38 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B2_A1_A1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0965443, 0.3969810, -0.0054744, 0.7100763, -0.6135321, 0.4024554
1: 0.0565891, 0.4885263, -0.0693125, 0.9573513, -0.9007622, 0.5578388
2: 0.0083284, 0.4923123, -0.1748363, 0.7621485, -0.7538201, 0.6671486
3: -0.0780025, 0.4490798, -0.2408400, 0.9047788, -0.9827812, 0.6899198
4: -0.0766807, 0.5934064, -0.2802821, 1.0476310, -1.1243117, 0.8736885

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B2_B1_B2_A1_A1_A1_A1_A1

### Relational analysis result of IS_A1_A2_B2_B1_B2_A1_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5683143, upper bound: 0.5109562
time: 0.42 seconds

## Relational analysis of IS_A1_A2_B2_B1_B2_A1_A1_A1_A1_A2

### Relational analysis result of IS_A1_A2_B2_B1_B2_A1_A1_A1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5645366, upper bound: 0.5109562
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_B1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0069308, 0.7141405, 0.0965443, 0.3969810, -0.4039118, 0.6175963
1: -0.0711197, 0.9631418, 0.0565891, 0.4885263, -0.5596460, 0.9065527
2: -0.1773483, 0.7663486, 0.0083284, 0.4923123, -0.6696606, 0.7580202
3: -0.2441539, 0.9109904, -0.0780025, 0.4490798, -0.6932336, 0.9889928
4: -0.2841340, 1.0538890, -0.0766807, 0.5934064, -0.8775404, 1.1305697

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_B2_A1_B1_B1_B1_B1_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_B1_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5111559, upper bound: 0.5684190
time: 0.37 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_B1_B1_B1_B1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_B1_B1_B1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5111559, upper bound: 0.5598047
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0433788, 0.4866467, 0.0873206, 0.4219245, -0.3785456, 0.3993261
1: -0.0069845, 0.5967795, 0.0395762, 0.5401090, -0.5470935, 0.5572033
2: -0.0581915, 0.5744838, -0.0181572, 0.5037278, -0.5619193, 0.5926410
3: -0.1468202, 0.5730036, -0.0989394, 0.4913468, -0.6381670, 0.6719431
4: -0.1378696, 0.6996939, -0.0983677, 0.6463114, -0.7841810, 0.7980616

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_B2_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5415877, upper bound: 0.5654935
time: 0.43 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5304913, upper bound: 0.5654935
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0283030, 0.5522523, 0.0873206, 0.4219245, -0.3936214, 0.4649317
1: -0.0250288, 0.7141986, 0.0395762, 0.5401090, -0.5651378, 0.6746224
2: -0.0988264, 0.6263493, -0.0181572, 0.5037278, -0.6025542, 0.6445066
3: -0.1721501, 0.6781709, -0.0989394, 0.4913468, -0.6634969, 0.7771103
4: -0.1945045, 0.8138232, -0.0983677, 0.6463114, -0.8408159, 0.9121909

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_B2_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5304913, upper bound: 0.5654935
time: 0.40 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5304913, upper bound: 0.5654935
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B1_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: 0.0522337, 0.4859760, 0.0873206, 0.4219245, -0.3696908, 0.3986554
1: 0.0017436, 0.5758268, 0.0395762, 0.5401090, -0.5383654, 0.5362506
2: -0.0330665, 0.5877808, -0.0181572, 0.5037278, -0.5367943, 0.6059381
3: -0.1326582, 0.5599031, -0.0989394, 0.4913468, -0.6240050, 0.6588426
4: -0.1118598, 0.6914989, -0.0983677, 0.6463114, -0.7581712, 0.7898666

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_B2_A2_B1_A2_B1_A1_A2_A1

### Relational analysis result of IS_A2_B1_B2_A2_B1_A2_B1_A1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5415877, upper bound: 0.5507922
time: 0.41 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_A2_B1_A1_A2_A2

### Relational analysis result of IS_A2_B1_B2_A2_B1_A2_B1_A1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5304913, upper bound: 0.5507922
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0233569, 0.5570214, 0.0873206, 0.4219245, -0.3985676, 0.4697009
1: -0.0314330, 0.7206672, 0.0395762, 0.5401090, -0.5715420, 0.6810911
2: -0.1071281, 0.6316106, -0.0181572, 0.5037278, -0.6108559, 0.6497678
3: -0.1827013, 0.6866775, -0.0989394, 0.4913468, -0.6740481, 0.7856169
4: -0.2049841, 0.8204701, -0.0983677, 0.6463114, -0.8512955, 0.9188378

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_B2_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_B2_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5304913, upper bound: 0.5654935
time: 0.39 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_B2_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5304913, upper bound: 0.5754925
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B2_B1_B1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0083361, 0.5719668, 0.0969255, 0.3960601, -0.3877240, 0.4750413
1: -0.0506135, 0.7433118, 0.0570602, 0.4867831, -0.5373967, 0.6862516
2: -0.1297052, 0.6455543, 0.0090892, 0.4917682, -0.6214734, 0.6364651
3: -0.2118729, 0.7126579, -0.0774024, 0.4474990, -0.6593719, 0.7900603
4: -0.2322460, 0.8428500, -0.0759368, 0.5922844, -0.8245305, 0.9187868

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B1_B1_B1_B1

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5797783
time: 0.41 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B1_B1_B1_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5754925
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B2_B1_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0083361, 0.5719668, 0.0925431, 0.3798722, -0.3715361, 0.4794236
1: -0.0506135, 0.7433118, 0.0508025, 0.4456694, -0.4962829, 0.6925094
2: -0.1297052, 0.6455543, 0.0190966, 0.4847844, -0.6144896, 0.6264577
3: -0.2118729, 0.7126579, -0.0695522, 0.4218645, -0.6337374, 0.7822101
4: -0.2322460, 0.8428500, -0.0605929, 0.5501102, -0.7823563, 0.9034429

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B1_B1_B2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5724233
time: 0.45 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B1_B1_B2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1_B1_B1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5611240
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B2_B1_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0274241, 0.5315738, 0.0937147, 0.4119777, -0.3845536, 0.4378591
1: -0.0291177, 0.6686592, 0.0489067, 0.5234078, -0.5525255, 0.6197525
2: -0.0887789, 0.6183543, -0.0067511, 0.4970919, -0.5858707, 0.6251054
3: -0.1754975, 0.6463004, -0.0869917, 0.4748205, -0.6503180, 0.7332922
4: -0.1838146, 0.7758477, -0.0889151, 0.6335514, -0.8173660, 0.8647628

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B1_B2_A1_A1

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1_B1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5464466
time: 0.40 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B1_B2_A1_A2

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1_B1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5608028
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B2_B1_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0149566, 0.5658405, 0.0937147, 0.4119777, -0.3970211, 0.4721258
1: -0.0463459, 0.7114239, 0.0489067, 0.5234078, -0.5697538, 0.6625172
2: -0.1064991, 0.6583920, -0.0067511, 0.4970919, -0.6035910, 0.6651430
3: -0.2061497, 0.6969915, -0.0869917, 0.4748205, -0.6809702, 0.7839832
4: -0.2113429, 0.8311484, -0.0889151, 0.6335514, -0.8448943, 0.9200635

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B1_B2_A2_A1

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1_B1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5507922
time: 0.41 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B1_B2_A2_A2

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1_B1_B2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5611240
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B2_B1_B2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0083361, 0.5719668, 0.0969255, 0.3960601, -0.3877240, 0.4750413
1: -0.0506135, 0.7433118, 0.0570602, 0.4867831, -0.5373967, 0.6862516
2: -0.1297052, 0.6455543, 0.0090892, 0.4917682, -0.6214734, 0.6364651
3: -0.2118729, 0.7126579, -0.0774024, 0.4474990, -0.6593719, 0.7900603
4: -0.2322460, 0.8428500, -0.0759368, 0.5922844, -0.8245305, 0.9187868

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B2_B1_B1_B1

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5797783
time: 0.43 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B2_B1_B1_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5754925
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B2_B1_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0083361, 0.5719668, 0.0921624, 0.3807045, -0.3723685, 0.4798044
1: -0.0506135, 0.7433118, 0.0501726, 0.4466347, -0.4972482, 0.6931393
2: -0.1297052, 0.6455543, 0.0185485, 0.4862787, -0.6159838, 0.6270058
3: -0.2118729, 0.7126579, -0.0701120, 0.4227494, -0.6346223, 0.7827699
4: -0.2322460, 0.8428500, -0.0608306, 0.5514524, -0.7836984, 0.9036806

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B2_B1_B2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5197859, upper bound: 0.5724233
time: 0.42 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B2_B1_B2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1_B2_B1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5197859, upper bound: 0.5611240
time: 0.39 seconds

## Summary of splitting at layer (split count: 9)
- Time for IS candidates: 2.70 seconds
IS_A1_A2_B2_B1_B1_A1_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 10, time: 2.70
Output dim: 0, lower bound: -0.5696524, upper bound: 0.4702711
IS_A1_A2_B2_B1_B1_A1_A1_A1_A1_A2, status: Status.UNKNOWN, split count: 10, time: 2.70
Output dim: 0, lower bound: -0.5694090, upper bound: 0.4702711
IS_A1_A2_B2_B1_B1_A1_A1_A2_A1_A1, status: Status.UNKNOWN, split count: 10, time: 2.70
Output dim: 0, lower bound: -0.5694090, upper bound: 0.4702711
IS_A1_A2_B2_B1_B1_A1_A1_A2_A1_A2, status: Status.UNKNOWN, split count: 10, time: 2.70
Output dim: 0, lower bound: -0.5694090, upper bound: 0.4702711
IS_A1_A2_B2_B1_B1_A2_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 10, time: 2.70
Output dim: 0, lower bound: -0.5663972, upper bound: 0.4605393
IS_A1_A2_B2_B1_B1_A2_A1_A1_A1_A2, status: Status.UNKNOWN, split count: 10, time: 2.70
Output dim: 0, lower bound: -0.5662008, upper bound: 0.4605393
IS_A1_A2_B2_B1_B1_A2_A1_A2_A1_A1, status: Status.UNKNOWN, split count: 10, time: 2.70
Output dim: 0, lower bound: -0.5674961, upper bound: 0.4893037
IS_A1_A2_B2_B1_B1_A2_A1_A2_A1_A2, status: Status.UNKNOWN, split count: 10, time: 2.70
Output dim: 0, lower bound: -0.5674961, upper bound: 0.5696298
IS_A1_A2_B2_B1_B1_A2_A1_A2_A2_A1, status: Status.VERIFIED, split count: 10, time: 2.70
Output dim: 0, lower bound: -0.5586579, upper bound: 0.5421828
IS_A1_A2_B2_B1_B1_A2_A1_A2_A2_A2, status: Status.VERIFIED, split count: 10, time: 2.70
Output dim: 0, lower bound: -0.5464140, upper bound: 0.5582596
IS_A1_A2_B2_B1_B2_A1_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 10, time: 2.70
Output dim: 0, lower bound: -0.5683143, upper bound: 0.5109562
IS_A1_A2_B2_B1_B2_A1_A1_A1_A1_A2, status: Status.VERIFIED, split count: 10, time: 2.70
Output dim: 0, lower bound: -0.5645366, upper bound: 0.5109562
IS_A2_B1_B2_A1_B1_B1_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 10, time: 2.70
Output dim: 0, lower bound: -0.5111559, upper bound: 0.5684190
IS_A2_B1_B2_A1_B1_B1_B1_B1_B1_B2, status: Status.VERIFIED, split count: 10, time: 2.70
Output dim: 0, lower bound: -0.5111559, upper bound: 0.5598047
IS_A2_B1_B2_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 10, time: 2.70
Output dim: 0, lower bound: -0.5415877, upper bound: 0.5654935
IS_A2_B1_B2_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 10, time: 2.70
Output dim: 0, lower bound: -0.5304913, upper bound: 0.5654935
IS_A2_B1_B2_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 10, time: 2.70
Output dim: 0, lower bound: -0.5304913, upper bound: 0.5654935
IS_A2_B1_B2_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 10, time: 2.70
Output dim: 0, lower bound: -0.5304913, upper bound: 0.5654935
IS_A2_B1_B2_A2_B1_A2_B1_A1_A2_A1, status: Status.VERIFIED, split count: 10, time: 2.70
Output dim: 0, lower bound: -0.5415877, upper bound: 0.5507922
IS_A2_B1_B2_A2_B1_A2_B1_A1_A2_A2, status: Status.VERIFIED, split count: 10, time: 2.70
Output dim: 0, lower bound: -0.5304913, upper bound: 0.5507922
IS_A2_B1_B2_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 10, time: 2.70
Output dim: 0, lower bound: -0.5304913, upper bound: 0.5654935
IS_A2_B1_B2_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 10, time: 2.70
Output dim: 0, lower bound: -0.5304913, upper bound: 0.5754925
IS_A2_B1_B2_A2_B2_B1_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 10, time: 2.70
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5797783
IS_A2_B1_B2_A2_B2_B1_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 10, time: 2.70
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5754925
IS_A2_B1_B2_A2_B2_B1_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 10, time: 2.70
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5724233
IS_A2_B1_B2_A2_B2_B1_B1_B1_B2_B2, status: Status.VERIFIED, split count: 10, time: 2.70
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5611240
IS_A2_B1_B2_A2_B2_B1_B1_B2_A1_A1, status: Status.VERIFIED, split count: 10, time: 2.70
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5464466
IS_A2_B1_B2_A2_B2_B1_B1_B2_A1_A2, status: Status.VERIFIED, split count: 10, time: 2.70
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5608028
IS_A2_B1_B2_A2_B2_B1_B1_B2_A2_A1, status: Status.VERIFIED, split count: 10, time: 2.70
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5507922
IS_A2_B1_B2_A2_B2_B1_B1_B2_A2_A2, status: Status.VERIFIED, split count: 10, time: 2.70
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5611240
IS_A2_B1_B2_A2_B2_B1_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 10, time: 2.70
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5797783
IS_A2_B1_B2_A2_B2_B1_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 10, time: 2.70
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5754925
IS_A2_B1_B2_A2_B2_B1_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 10, time: 2.70
Output dim: 0, lower bound: -0.5197859, upper bound: 0.5724233
IS_A2_B1_B2_A2_B2_B1_B2_B1_B2_B2, status: Status.VERIFIED, split count: 10, time: 2.70
Output dim: 0, lower bound: -0.5197859, upper bound: 0.5611240

## BFS IS instance: IS_A1_A2_B2_B1_B1_A1_A1_A1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0945764, 0.3505142, 0.0646830, 0.5288626, -0.4342862, 0.2845256
1: 0.0685086, 0.3967934, 0.0195227, 0.7196292, -0.6511206, 0.3772707
2: 0.0585651, 0.4334365, -0.0684017, 0.5826666, -0.5241014, 0.5018382
3: -0.0406969, 0.4090275, -0.1308922, 0.6449826, -0.6856795, 0.5399196
4: -0.0057726, 0.4402502, -0.1473095, 0.8047376, -0.8105102, 0.5875597

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B2_B1_B1_A1_A1_A1_A1_A1_A1

### Relational analysis result of IS_A1_A2_B2_B1_B1_A1_A1_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5663972, upper bound: 0.4605393
time: 0.42 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_A1_A1_A1_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B2_B1_B1_A1_A1_A1_A1_A1_A1

### Relational analysis result of IS_A1_A2_B2_B1_B1_A1_A1_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5696524, upper bound: 0.4702711
time: 0.41 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_A1_A1_A1_A1_A1_A2

### Relational analysis result of IS_A1_A2_B2_B1_B1_A1_A1_A1_A1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5617467, upper bound: 0.4702711
time: 0.37 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B1_A1_A1_A1_A1_A2

### Backsubstitution after applying IS history:
0: 0.0924387, 0.3566364, 0.0644008, 0.5302621, -0.4378234, 0.2922356
1: 0.0503964, 0.4023641, 0.0192177, 0.7222201, -0.6718237, 0.3831464
2: 0.0225284, 0.5009353, -0.0694884, 0.5835138, -0.5609854, 0.5704237
3: -0.0864439, 0.3917682, -0.1317451, 0.6470909, -0.7335348, 0.5235133
4: -0.0891201, 0.5688827, -0.1483382, 0.8069319, -0.8960520, 0.7172209

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B2_B1_B1_A1_A1_A1_A1_A2_A1

### Relational analysis result of IS_A1_A2_B2_B1_B1_A1_A1_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5694090, upper bound: 0.4702711
time: 0.40 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_A1_A1_A1_A1_A2_A2

### Relational analysis result of IS_A1_A2_B2_B1_B1_A1_A1_A1_A1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5613755, upper bound: 0.4702711
time: 0.41 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B1_A1_A1_A2_A1_A1

### Backsubstitution after applying IS history:
0: 0.0947785, 0.3503857, 0.0646830, 0.5288626, -0.4340841, 0.2844018
1: 0.0685635, 0.3965145, 0.0195227, 0.7196292, -0.6510658, 0.3769919
2: 0.0587087, 0.4331901, -0.0684017, 0.5826666, -0.5239579, 0.5015918
3: -0.0405765, 0.4087051, -0.1308922, 0.6449826, -0.6855591, 0.5395973
4: -0.0056551, 0.4400306, -0.1473095, 0.8047376, -0.8103926, 0.5873401

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B2_B1_B1_A1_A1_A2_A1_A1_A1

### Relational analysis result of IS_A1_A2_B2_B1_B1_A1_A1_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5663972, upper bound: 0.4605393
time: 0.42 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_A1_A1_A2_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B2_B1_B1_A1_A1_A2_A1_A1_A1

### Relational analysis result of IS_A1_A2_B2_B1_B1_A1_A1_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5696524, upper bound: 0.4702711
time: 0.41 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_A1_A1_A2_A1_A1_A2

### Relational analysis result of IS_A1_A2_B2_B1_B1_A1_A1_A2_A1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5617467, upper bound: 0.4702711
time: 0.38 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B1_A1_A1_A2_A1_A2

### Backsubstitution after applying IS history:
0: 0.0924387, 0.3566364, 0.0644008, 0.5302621, -0.4378234, 0.2922356
1: 0.0503964, 0.4023641, 0.0192177, 0.7222201, -0.6718237, 0.3831464
2: 0.0225284, 0.5009353, -0.0694884, 0.5835138, -0.5609854, 0.5704237
3: -0.0864439, 0.3917682, -0.1317451, 0.6470909, -0.7335348, 0.5235133
4: -0.0891201, 0.5688827, -0.1483382, 0.8069319, -0.8960520, 0.7172209

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B2_B1_B1_A1_A1_A2_A1_A2_A1

### Relational analysis result of IS_A1_A2_B2_B1_B1_A1_A1_A2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5694090, upper bound: 0.4702711
time: 0.41 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_A1_A1_A2_A1_A2_A2

### Relational analysis result of IS_A1_A2_B2_B1_B1_A1_A1_A2_A1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5613755, upper bound: 0.4702711
time: 0.41 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B1_A2_A1_A1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0998964, 0.3467079, 0.0646830, 0.5288626, -0.4289662, 0.2820250
1: 0.0816488, 0.3923656, 0.0195227, 0.7196292, -0.6379804, 0.3728429
2: 0.0676711, 0.4236782, -0.0684017, 0.5826666, -0.5149955, 0.4920799
3: -0.0211008, 0.4035573, -0.1308922, 0.6449826, -0.6660834, 0.5344495
4: 0.0067785, 0.4217074, -0.1473095, 0.8047376, -0.7979591, 0.5690169

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B2_B1_B1_A2_A1_A1_A1_A1_A1

### Relational analysis result of IS_A1_A2_B2_B1_B1_A2_A1_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5663972, upper bound: 0.4605393
time: 0.43 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_A2_A1_A1_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B2_B1_B1_A2_A1_A1_A1_A1_A1

### Relational analysis result of IS_A1_A2_B2_B1_B1_A2_A1_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5663972, upper bound: 0.4605393
time: 0.42 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_A2_A1_A1_A1_A1_A2

### Relational analysis result of IS_A1_A2_B2_B1_B1_A2_A1_A1_A1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5583578, upper bound: 0.4605393
time: 0.40 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B1_A2_A1_A1_A1_A2

### Backsubstitution after applying IS history:
0: 0.1012268, 0.3140154, 0.0644008, 0.5302621, -0.4290353, 0.2496146
1: 0.0622427, 0.3587341, 0.0192177, 0.7222201, -0.6599774, 0.3395164
2: 0.0383472, 0.4491130, -0.0694884, 0.5835138, -0.5451666, 0.5186014
3: -0.0713305, 0.3488707, -0.1317451, 0.6470909, -0.7184215, 0.4806158
4: -0.0695770, 0.5056835, -0.1483382, 0.8069319, -0.8765088, 0.6540217

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B2_B1_B1_A2_A1_A1_A1_A2_A1

### Relational analysis result of IS_A1_A2_B2_B1_B1_A2_A1_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5662008, upper bound: 0.4605393
time: 0.40 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_A2_A1_A1_A1_A2_A2

### Relational analysis result of IS_A1_A2_B2_B1_B1_A2_A1_A1_A1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5580973, upper bound: 0.4605393
time: 0.41 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B1_A2_A1_A2_A1_A1

### Backsubstitution after applying IS history:
0: 0.0935934, 0.3446476, 0.0646830, 0.5288626, -0.4352692, 0.2799646
1: 0.0695772, 0.3865252, 0.0195227, 0.7196292, -0.6500520, 0.3670025
2: 0.0587194, 0.4287091, -0.0684017, 0.5826666, -0.5239472, 0.4971108
3: -0.0373166, 0.3964920, -0.1308922, 0.6449826, -0.6822992, 0.5273842
4: -0.0043771, 0.4454577, -0.1473095, 0.8047376, -0.8091147, 0.5927672

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B2_B1_B1_A2_A1_A2_A1_A1_A1

### Relational analysis result of IS_A1_A2_B2_B1_B1_A2_A1_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5677527, upper bound: 0.4893037
time: 0.43 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_A2_A1_A2_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B2_B1_B1_A2_A1_A2_A1_A1_A1

### Relational analysis result of IS_A1_A2_B2_B1_B1_A2_A1_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5663972, upper bound: 0.4605393
time: 0.44 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_A2_A1_A2_A1_A1_A2

### Relational analysis result of IS_A1_A2_B2_B1_B1_A2_A1_A2_A1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5583578, upper bound: 0.4779336
time: 0.43 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B1_A2_A1_A2_A1_A2

### Backsubstitution after applying IS history:
0: 0.0880811, 0.4212151, 0.0644008, 0.5302621, -0.4421810, 0.3568143
1: 0.0402591, 0.5388805, 0.0192177, 0.7222201, -0.6819610, 0.5196627
2: -0.0172656, 0.5033177, -0.0694884, 0.5835138, -0.6007794, 0.5728061
3: -0.0979600, 0.4902630, -0.1317451, 0.6470909, -0.7450509, 0.6220081
4: -0.0975381, 0.6454344, -0.1483382, 0.8069319, -0.9044700, 0.7937726

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B2_B1_B1_A2_A1_A2_A1_A2_A1

### Relational analysis result of IS_A1_A2_B2_B1_B1_A2_A1_A2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5674961, upper bound: 0.5696298
time: 0.44 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_A2_A1_A2_A1_A2_A2

### Relational analysis result of IS_A1_A2_B2_B1_B1_A2_A1_A2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5595722, upper bound: 0.5696298
time: 0.41 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B2_A1_A1_A1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0965443, 0.3969810, -0.0054744, 0.7100763, -0.6135321, 0.4024554
1: 0.0565891, 0.4885263, -0.0693125, 0.9573513, -0.9007622, 0.5578388
2: 0.0083284, 0.4923123, -0.1748363, 0.7621485, -0.7538201, 0.6671486
3: -0.0780025, 0.4490798, -0.2408400, 0.9047788, -0.9827812, 0.6899198
4: -0.0766807, 0.5934064, -0.2802821, 1.0476310, -1.1243117, 0.8736885

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B2_B1_B2_A1_A1_A1_A1_A1_A1

### Relational analysis result of IS_A1_A2_B2_B1_B2_A1_A1_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5683143, upper bound: 0.5109562
time: 0.43 seconds

## Relational analysis of IS_A1_A2_B2_B1_B2_A1_A1_A1_A1_A1_A2

### Relational analysis result of IS_A1_A2_B2_B1_B2_A1_A1_A1_A1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5596943, upper bound: 0.5109562
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_B1_B1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0069308, 0.7141405, 0.0965443, 0.3969810, -0.4039118, 0.6175963
1: -0.0711197, 0.9631418, 0.0565891, 0.4885263, -0.5596460, 0.9065527
2: -0.1773483, 0.7663486, 0.0083284, 0.4923123, -0.6696606, 0.7580202
3: -0.2441539, 0.9109904, -0.0780025, 0.4490798, -0.6932336, 0.9889928
4: -0.2841340, 1.0538890, -0.0766807, 0.5934064, -0.8775404, 1.1305697

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_B2_A1_B1_B1_B1_B1_B1_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_B1_B1_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5111559, upper bound: 0.5684190
time: 0.40 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_B1_B1_B1_B1_B1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_B1_B1_B1_B1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5111559, upper bound: 0.5646490
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0433788, 0.4866467, 0.0873206, 0.4219245, -0.3785456, 0.3993261
1: -0.0069845, 0.5967795, 0.0395762, 0.5401090, -0.5470935, 0.5572033
2: -0.0581915, 0.5744838, -0.0181572, 0.5037278, -0.5619193, 0.5926410
3: -0.1468202, 0.5730036, -0.0989394, 0.4913468, -0.6381670, 0.6719431
4: -0.1378696, 0.6996939, -0.0983677, 0.6463114, -0.7841810, 0.7980616

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_B2_A2_B1_A1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_B2_A2_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5415877, upper bound: 0.5654935
time: 0.45 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5415877, upper bound: 0.5507922
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0283030, 0.5522523, 0.0873206, 0.4219245, -0.3936214, 0.4649317
1: -0.0250288, 0.7141986, 0.0395762, 0.5401090, -0.5651378, 0.6746224
2: -0.0988264, 0.6263493, -0.0181572, 0.5037278, -0.6025542, 0.6445066
3: -0.1721501, 0.6781709, -0.0989394, 0.4913468, -0.6634969, 0.7771103
4: -0.1945045, 0.8138232, -0.0983677, 0.6463114, -0.8408159, 0.9121909

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_B2_A2_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5304913, upper bound: 0.5654935
time: 0.41 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5304913, upper bound: 0.5507922
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0433788, 0.4866467, 0.0873206, 0.4219245, -0.3785456, 0.3993261
1: -0.0069845, 0.5967795, 0.0395762, 0.5401090, -0.5470935, 0.5572033
2: -0.0581915, 0.5744838, -0.0181572, 0.5037278, -0.5619193, 0.5926410
3: -0.1468202, 0.5730036, -0.0989394, 0.4913468, -0.6381670, 0.6719431
4: -0.1378696, 0.6996939, -0.0983677, 0.6463114, -0.7841810, 0.7980616

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_B2_A2_B1_A1_B1_A2_B1_A1_A1

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1_B1_A2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5415877, upper bound: 0.5653172
time: 0.46 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_A1_B1_A2_B1_A1_A2

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1_B1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5415877, upper bound: 0.5654935
time: 0.45 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0283030, 0.5522523, 0.0873206, 0.4219245, -0.3936214, 0.4649317
1: -0.0250288, 0.7141986, 0.0395762, 0.5401090, -0.5651378, 0.6746224
2: -0.0988264, 0.6263493, -0.0181572, 0.5037278, -0.6025542, 0.6445066
3: -0.1721501, 0.6781709, -0.0989394, 0.4913468, -0.6634969, 0.7771103
4: -0.1945045, 0.8138232, -0.0983677, 0.6463114, -0.8408159, 0.9121909

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_B2_A2_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5304913, upper bound: 0.5654935
time: 0.41 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5304913, upper bound: 0.5507922
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0433788, 0.4866467, 0.0873206, 0.4219245, -0.3785456, 0.3993261
1: -0.0069845, 0.5967795, 0.0395762, 0.5401090, -0.5470935, 0.5572033
2: -0.0581915, 0.5744838, -0.0181572, 0.5037278, -0.5619193, 0.5926410
3: -0.1468202, 0.5730036, -0.0989394, 0.4913468, -0.6381670, 0.6719431
4: -0.1378696, 0.6996939, -0.0983677, 0.6463114, -0.7841810, 0.7980616

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_B2_A2_B1_A2_B1_A2_B1_A1_A1

### Relational analysis result of IS_A2_B1_B2_A2_B1_A2_B1_A2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5415877, upper bound: 0.5653172
time: 0.45 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_A2_B1_A2_B1_A1_A2

### Relational analysis result of IS_A2_B1_B2_A2_B1_A2_B1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5415877, upper bound: 0.5654935
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0233569, 0.5570214, 0.0873206, 0.4219245, -0.3985676, 0.4697009
1: -0.0314330, 0.7206672, 0.0395762, 0.5401090, -0.5715420, 0.6810911
2: -0.1071281, 0.6316106, -0.0181572, 0.5037278, -0.6108559, 0.6497678
3: -0.1827013, 0.6866775, -0.0989394, 0.4913468, -0.6740481, 0.7856169
4: -0.2049841, 0.8204701, -0.0983677, 0.6463114, -0.8512955, 0.9188378

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_B2_A2_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5304913, upper bound: 0.5754925
time: 0.46 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B1_A2_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5247494, upper bound: 0.5611240
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B2_B1_B1_B1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0083361, 0.5719668, 0.0969255, 0.3960601, -0.3877240, 0.4750413
1: -0.0506135, 0.7433118, 0.0570602, 0.4867831, -0.5373967, 0.6862516
2: -0.1297052, 0.6455543, 0.0090892, 0.4917682, -0.6214734, 0.6364651
3: -0.2118729, 0.7126579, -0.0774024, 0.4474990, -0.6593719, 0.7900603
4: -0.2322460, 0.8428500, -0.0759368, 0.5922844, -0.8245305, 0.9187868

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B1_B1_B1_B1_B1

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1_B1_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5797783
time: 0.45 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B1_B1_B1_B1_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1_B1_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5724233
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B2_B1_B1_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0083361, 0.5719668, 0.0937147, 0.4119777, -0.4036416, 0.4782521
1: -0.0506135, 0.7433118, 0.0489067, 0.5234078, -0.5740213, 0.6944051
2: -0.1297052, 0.6455543, -0.0067511, 0.4970919, -0.6267971, 0.6523054
3: -0.2118729, 0.7126579, -0.0869917, 0.4748205, -0.6866934, 0.7996497
4: -0.2322460, 0.8428500, -0.0889151, 0.6335514, -0.8657974, 0.9317651

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B1_B1_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1_B1_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5754925
time: 0.41 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B1_B1_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1_B1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5754925
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B2_B1_B1_B1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0083361, 0.5719668, 0.0925431, 0.3798722, -0.3715361, 0.4794236
1: -0.0506135, 0.7433118, 0.0508025, 0.4456694, -0.4962829, 0.6925094
2: -0.1297052, 0.6455543, 0.0190966, 0.4847844, -0.6144896, 0.6264577
3: -0.2118729, 0.7126579, -0.0695522, 0.4218645, -0.6337374, 0.7822101
4: -0.2322460, 0.8428500, -0.0605929, 0.5501102, -0.7823563, 0.9034429

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B1_B1_B2_B1_B1

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1_B1_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5724233
time: 0.47 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B1_B1_B2_B1_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1_B1_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5724233
time: 0.46 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B2_B1_B2_B1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0083361, 0.5719668, 0.0969255, 0.3960601, -0.3877240, 0.4750413
1: -0.0506135, 0.7433118, 0.0570602, 0.4867831, -0.5373967, 0.6862516
2: -0.1297052, 0.6455543, 0.0090892, 0.4917682, -0.6214734, 0.6364651
3: -0.2118729, 0.7126579, -0.0774024, 0.4474990, -0.6593719, 0.7900603
4: -0.2322460, 0.8428500, -0.0759368, 0.5922844, -0.8245305, 0.9187868

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B2_B1_B1_B1_B1

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1_B2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5797783
time: 0.45 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B2_B1_B1_B1_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1_B2_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5724233
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B2_B1_B2_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0083361, 0.5719668, 0.0937147, 0.4119777, -0.4036416, 0.4782521
1: -0.0506135, 0.7433118, 0.0489067, 0.5234078, -0.5740213, 0.6944051
2: -0.1297052, 0.6455543, -0.0067511, 0.4970919, -0.6267971, 0.6523054
3: -0.2118729, 0.7126579, -0.0869917, 0.4748205, -0.6866934, 0.7996497
4: -0.2322460, 0.8428500, -0.0889151, 0.6335514, -0.8657974, 0.9317651

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B2_B1_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1_B2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5754925
time: 0.41 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B2_B1_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1_B2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5754925
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B2_B1_B2_B1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0083361, 0.5719668, 0.0921624, 0.3807045, -0.3723685, 0.4798044
1: -0.0506135, 0.7433118, 0.0501726, 0.4466347, -0.4972482, 0.6931393
2: -0.1297052, 0.6455543, 0.0185485, 0.4862787, -0.6159838, 0.6270058
3: -0.2118729, 0.7126579, -0.0701120, 0.4227494, -0.6346223, 0.7827699
4: -0.2322460, 0.8428500, -0.0608306, 0.5514524, -0.7836984, 0.9036806

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B2_B1_B2_B1_B1

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1_B2_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5724233
time: 0.44 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B2_B1_B2_B1_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1_B2_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5197859, upper bound: 0.5724233
time: 0.41 seconds

## Summary of splitting at layer (split count: 10)
- Time for IS candidates: 2.85 seconds
IS_A1_A2_B2_B1_B1_A1_A1_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 11, time: 2.85
Output dim: 0, lower bound: -0.5696524, upper bound: 0.4702711
IS_A1_A2_B2_B1_B1_A1_A1_A1_A1_A1_A2, status: Status.VERIFIED, split count: 11, time: 2.85
Output dim: 0, lower bound: -0.5617467, upper bound: 0.4702711
IS_A1_A2_B2_B1_B1_A1_A1_A1_A1_A2_A1, status: Status.UNKNOWN, split count: 11, time: 2.85
Output dim: 0, lower bound: -0.5694090, upper bound: 0.4702711
IS_A1_A2_B2_B1_B1_A1_A1_A1_A1_A2_A2, status: Status.VERIFIED, split count: 11, time: 2.85
Output dim: 0, lower bound: -0.5613755, upper bound: 0.4702711
IS_A1_A2_B2_B1_B1_A1_A1_A2_A1_A1_A1, status: Status.UNKNOWN, split count: 11, time: 2.85
Output dim: 0, lower bound: -0.5696524, upper bound: 0.4702711
IS_A1_A2_B2_B1_B1_A1_A1_A2_A1_A1_A2, status: Status.VERIFIED, split count: 11, time: 2.85
Output dim: 0, lower bound: -0.5617467, upper bound: 0.4702711
IS_A1_A2_B2_B1_B1_A1_A1_A2_A1_A2_A1, status: Status.UNKNOWN, split count: 11, time: 2.85
Output dim: 0, lower bound: -0.5694090, upper bound: 0.4702711
IS_A1_A2_B2_B1_B1_A1_A1_A2_A1_A2_A2, status: Status.VERIFIED, split count: 11, time: 2.85
Output dim: 0, lower bound: -0.5613755, upper bound: 0.4702711
IS_A1_A2_B2_B1_B1_A2_A1_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 11, time: 2.85
Output dim: 0, lower bound: -0.5663972, upper bound: 0.4605393
IS_A1_A2_B2_B1_B1_A2_A1_A1_A1_A1_A2, status: Status.VERIFIED, split count: 11, time: 2.85
Output dim: 0, lower bound: -0.5583578, upper bound: 0.4605393
IS_A1_A2_B2_B1_B1_A2_A1_A1_A1_A2_A1, status: Status.UNKNOWN, split count: 11, time: 2.85
Output dim: 0, lower bound: -0.5662008, upper bound: 0.4605393
IS_A1_A2_B2_B1_B1_A2_A1_A1_A1_A2_A2, status: Status.VERIFIED, split count: 11, time: 2.85
Output dim: 0, lower bound: -0.5580973, upper bound: 0.4605393
IS_A1_A2_B2_B1_B1_A2_A1_A2_A1_A1_A1, status: Status.UNKNOWN, split count: 11, time: 2.85
Output dim: 0, lower bound: -0.5663972, upper bound: 0.4605393
IS_A1_A2_B2_B1_B1_A2_A1_A2_A1_A1_A2, status: Status.VERIFIED, split count: 11, time: 2.85
Output dim: 0, lower bound: -0.5583578, upper bound: 0.4779336
IS_A1_A2_B2_B1_B1_A2_A1_A2_A1_A2_A1, status: Status.UNKNOWN, split count: 11, time: 2.85
Output dim: 0, lower bound: -0.5674961, upper bound: 0.5696298
IS_A1_A2_B2_B1_B1_A2_A1_A2_A1_A2_A2, status: Status.UNKNOWN, split count: 11, time: 2.85
Output dim: 0, lower bound: -0.5595722, upper bound: 0.5696298
IS_A1_A2_B2_B1_B2_A1_A1_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 11, time: 2.85
Output dim: 0, lower bound: -0.5683143, upper bound: 0.5109562
IS_A1_A2_B2_B1_B2_A1_A1_A1_A1_A1_A2, status: Status.VERIFIED, split count: 11, time: 2.85
Output dim: 0, lower bound: -0.5596943, upper bound: 0.5109562
IS_A2_B1_B2_A1_B1_B1_B1_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 11, time: 2.85
Output dim: 0, lower bound: -0.5111559, upper bound: 0.5684190
IS_A2_B1_B2_A1_B1_B1_B1_B1_B1_B1_B2, status: Status.VERIFIED, split count: 11, time: 2.85
Output dim: 0, lower bound: -0.5111559, upper bound: 0.5646490
IS_A2_B1_B2_A2_B1_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 11, time: 2.85
Output dim: 0, lower bound: -0.5415877, upper bound: 0.5654935
IS_A2_B1_B2_A2_B1_A1_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 11, time: 2.85
Output dim: 0, lower bound: -0.5415877, upper bound: 0.5507922
IS_A2_B1_B2_A2_B1_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 11, time: 2.85
Output dim: 0, lower bound: -0.5304913, upper bound: 0.5654935
IS_A2_B1_B2_A2_B1_A1_B1_A1_B1_A2_B2, status: Status.VERIFIED, split count: 11, time: 2.85
Output dim: 0, lower bound: -0.5304913, upper bound: 0.5507922
IS_A2_B1_B2_A2_B1_A1_B1_A2_B1_A1_A1, status: Status.VERIFIED, split count: 11, time: 2.85
Output dim: 0, lower bound: -0.5415877, upper bound: 0.5653172
IS_A2_B1_B2_A2_B1_A1_B1_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 11, time: 2.85
Output dim: 0, lower bound: -0.5415877, upper bound: 0.5654935
IS_A2_B1_B2_A2_B1_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 11, time: 2.85
Output dim: 0, lower bound: -0.5304913, upper bound: 0.5654935
IS_A2_B1_B2_A2_B1_A1_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 11, time: 2.85
Output dim: 0, lower bound: -0.5304913, upper bound: 0.5507922
IS_A2_B1_B2_A2_B1_A2_B1_A2_B1_A1_A1, status: Status.VERIFIED, split count: 11, time: 2.85
Output dim: 0, lower bound: -0.5415877, upper bound: 0.5653172
IS_A2_B1_B2_A2_B1_A2_B1_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 11, time: 2.85
Output dim: 0, lower bound: -0.5415877, upper bound: 0.5654935
IS_A2_B1_B2_A2_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 11, time: 2.85
Output dim: 0, lower bound: -0.5304913, upper bound: 0.5754925
IS_A2_B1_B2_A2_B1_A2_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 11, time: 2.85
Output dim: 0, lower bound: -0.5247494, upper bound: 0.5611240
IS_A2_B1_B2_A2_B2_B1_B1_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 11, time: 2.85
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5797783
IS_A2_B1_B2_A2_B2_B1_B1_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 11, time: 2.85
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5724233
IS_A2_B1_B2_A2_B2_B1_B1_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 11, time: 2.85
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5754925
IS_A2_B1_B2_A2_B2_B1_B1_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 11, time: 2.85
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5754925
IS_A2_B1_B2_A2_B2_B1_B1_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 11, time: 2.85
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5724233
IS_A2_B1_B2_A2_B2_B1_B1_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 11, time: 2.85
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5724233
IS_A2_B1_B2_A2_B2_B1_B2_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 11, time: 2.85
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5797783
IS_A2_B1_B2_A2_B2_B1_B2_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 11, time: 2.85
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5724233
IS_A2_B1_B2_A2_B2_B1_B2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 11, time: 2.85
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5754925
IS_A2_B1_B2_A2_B2_B1_B2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 11, time: 2.85
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5754925
IS_A2_B1_B2_A2_B2_B1_B2_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 11, time: 2.85
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5724233
IS_A2_B1_B2_A2_B2_B1_B2_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 11, time: 2.85
Output dim: 0, lower bound: -0.5197859, upper bound: 0.5724233

## BFS IS instance: IS_A1_A2_B2_B1_B1_A1_A1_A1_A1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0945764, 0.3505142, 0.0644008, 0.5302621, -0.4356858, 0.2848256
1: 0.0685086, 0.3967934, 0.0192177, 0.7222201, -0.6537115, 0.3775756
2: 0.0585651, 0.4334365, -0.0694884, 0.5835138, -0.5249487, 0.5029249
3: -0.0406969, 0.4090275, -0.1317451, 0.6470909, -0.6877878, 0.5407726
4: -0.0057726, 0.4402502, -0.1483382, 0.8069319, -0.8127044, 0.5885884

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B2_B1_B1_A1_A1_A1_A1_A1_A1_A1

### Relational analysis result of IS_A1_A2_B2_B1_B1_A1_A1_A1_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5696524, upper bound: 0.4702711
time: 0.42 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_A1_A1_A1_A1_A1_A1_A2

### Relational analysis result of IS_A1_A2_B2_B1_B1_A1_A1_A1_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5694090, upper bound: 0.4702711
time: 0.39 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B1_A1_A1_A1_A1_A2_A1

### Backsubstitution after applying IS history:
0: 0.0924387, 0.3566364, 0.0644008, 0.5302621, -0.4378234, 0.2922356
1: 0.0503964, 0.4023641, 0.0192177, 0.7222201, -0.6718237, 0.3831464
2: 0.0225284, 0.5009353, -0.0694884, 0.5835138, -0.5609854, 0.5704237
3: -0.0864439, 0.3917682, -0.1317451, 0.6470909, -0.7335348, 0.5235133
4: -0.0891201, 0.5688827, -0.1483382, 0.8069319, -0.8960520, 0.7172209

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B2_B1_B1_A1_A1_A1_A1_A2_A1_A1

### Relational analysis result of IS_A1_A2_B2_B1_B1_A1_A1_A1_A1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5694090, upper bound: 0.4702711
time: 0.39 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_A1_A1_A1_A1_A2_A1_A2

### Relational analysis result of IS_A1_A2_B2_B1_B1_A1_A1_A1_A1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5694090, upper bound: 0.4702711
time: 0.40 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B1_A1_A1_A2_A1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0947785, 0.3503857, 0.0644008, 0.5302621, -0.4354836, 0.2847018
1: 0.0685635, 0.3965145, 0.0192177, 0.7222201, -0.6536567, 0.3772968
2: 0.0587087, 0.4331901, -0.0694884, 0.5835138, -0.5248051, 0.5026785
3: -0.0405765, 0.4087051, -0.1317451, 0.6470909, -0.6876674, 0.5404502
4: -0.0056551, 0.4400306, -0.1483382, 0.8069319, -0.8125869, 0.5883688

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B2_B1_B1_A1_A1_A2_A1_A1_A1_A1

### Relational analysis result of IS_A1_A2_B2_B1_B1_A1_A1_A2_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5696524, upper bound: 0.4702711
time: 0.42 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_A1_A1_A2_A1_A1_A1_A2

### Relational analysis result of IS_A1_A2_B2_B1_B1_A1_A1_A2_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5694090, upper bound: 0.4702711
time: 0.39 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B1_A1_A1_A2_A1_A2_A1

### Backsubstitution after applying IS history:
0: 0.0924387, 0.3566364, 0.0644008, 0.5302621, -0.4378234, 0.2922356
1: 0.0503964, 0.4023641, 0.0192177, 0.7222201, -0.6718237, 0.3831464
2: 0.0225284, 0.5009353, -0.0694884, 0.5835138, -0.5609854, 0.5704237
3: -0.0864439, 0.3917682, -0.1317451, 0.6470909, -0.7335348, 0.5235133
4: -0.0891201, 0.5688827, -0.1483382, 0.8069319, -0.8960520, 0.7172209

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B2_B1_B1_A1_A1_A2_A1_A2_A1_A1

### Relational analysis result of IS_A1_A2_B2_B1_B1_A1_A1_A2_A1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5694090, upper bound: 0.4702711
time: 0.40 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_A1_A1_A2_A1_A2_A1_A2

### Relational analysis result of IS_A1_A2_B2_B1_B1_A1_A1_A2_A1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5694090, upper bound: 0.4702711
time: 0.40 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B1_A2_A1_A1_A1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0998964, 0.3467079, 0.0644008, 0.5302621, -0.4303657, 0.2823071
1: 0.0816488, 0.3923656, 0.0192177, 0.7222201, -0.6405713, 0.3731478
2: 0.0676711, 0.4236782, -0.0694884, 0.5835138, -0.5158427, 0.4931666
3: -0.0211008, 0.4035573, -0.1317451, 0.6470909, -0.6681917, 0.5353024
4: 0.0067785, 0.4217074, -0.1483382, 0.8069319, -0.8001534, 0.5700456

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B2_B1_B1_A2_A1_A1_A1_A1_A1_A1

### Relational analysis result of IS_A1_A2_B2_B1_B1_A2_A1_A1_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5663972, upper bound: 0.4605393
time: 0.45 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_A2_A1_A1_A1_A1_A1_A2

### Relational analysis result of IS_A1_A2_B2_B1_B1_A2_A1_A1_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5662008, upper bound: 0.4605393
time: 0.41 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B1_A2_A1_A1_A1_A2_A1

### Backsubstitution after applying IS history:
0: 0.1012268, 0.3140154, 0.0644008, 0.5302621, -0.4290353, 0.2496146
1: 0.0622427, 0.3587341, 0.0192177, 0.7222201, -0.6599774, 0.3395164
2: 0.0383472, 0.4491130, -0.0694884, 0.5835138, -0.5451666, 0.5186014
3: -0.0713305, 0.3488707, -0.1317451, 0.6470909, -0.7184215, 0.4806158
4: -0.0695770, 0.5056835, -0.1483382, 0.8069319, -0.8765088, 0.6540217

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B2_B1_B1_A2_A1_A1_A1_A2_A1_A1

### Relational analysis result of IS_A1_A2_B2_B1_B1_A2_A1_A1_A1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5662008, upper bound: 0.4605393
time: 0.41 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_A2_A1_A1_A1_A2_A1_A2

### Relational analysis result of IS_A1_A2_B2_B1_B1_A2_A1_A1_A1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5662008, upper bound: 0.4605393
time: 0.41 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B1_A2_A1_A2_A1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0998964, 0.3467079, 0.0644008, 0.5302621, -0.4303657, 0.2823071
1: 0.0816488, 0.3923656, 0.0192177, 0.7222201, -0.6405713, 0.3731478
2: 0.0676711, 0.4236782, -0.0694884, 0.5835138, -0.5158427, 0.4931666
3: -0.0211008, 0.4035573, -0.1317451, 0.6470909, -0.6681917, 0.5353024
4: 0.0067785, 0.4217074, -0.1483382, 0.8069319, -0.8001534, 0.5700456

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B2_B1_B1_A2_A1_A2_A1_A1_A1_A1

### Relational analysis result of IS_A1_A2_B2_B1_B1_A2_A1_A2_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5663972, upper bound: 0.4605393
time: 0.44 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_A2_A1_A2_A1_A1_A1_A2

### Relational analysis result of IS_A1_A2_B2_B1_B1_A2_A1_A2_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5662008, upper bound: 0.4605393
time: 0.42 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B1_A2_A1_A2_A1_A2_A1

### Backsubstitution after applying IS history:
0: 0.0880811, 0.4212151, 0.0644008, 0.5302621, -0.4421810, 0.3568143
1: 0.0402591, 0.5388805, 0.0192177, 0.7222201, -0.6819610, 0.5196627
2: -0.0172656, 0.5033177, -0.0694884, 0.5835138, -0.6007794, 0.5728061
3: -0.0979600, 0.4902630, -0.1317451, 0.6470909, -0.7450509, 0.6220081
4: -0.0975381, 0.6454344, -0.1483382, 0.8069319, -0.9044700, 0.7937726

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B2_B1_B1_A2_A1_A2_A1_A2_A1_A1

### Relational analysis result of IS_A1_A2_B2_B1_B1_A2_A1_A2_A1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5674961, upper bound: 0.4893037
time: 0.48 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_A2_A1_A2_A1_A2_A1_A2

### Relational analysis result of IS_A1_A2_B2_B1_B1_A2_A1_A2_A1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5674961, upper bound: 0.5696298
time: 0.44 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B1_A2_A1_A2_A1_A2_A2

### Backsubstitution after applying IS history:
0: 0.0845213, 0.3948262, 0.0644008, 0.5302621, -0.4457408, 0.3304254
1: 0.0383507, 0.4757941, 0.0192177, 0.7222201, -0.6838694, 0.4565763
2: 0.0013437, 0.4952822, -0.0694884, 0.5835138, -0.5821701, 0.5647706
3: -0.0842046, 0.4470655, -0.1317451, 0.6470909, -0.7312955, 0.5788106
4: -0.0750232, 0.5913191, -0.1483382, 0.8069319, -0.8819550, 0.7396573

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B2_B1_B1_A2_A1_A2_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B2_B1_B1_A2_A1_A2_A1_A2_A2_A1

### Relational analysis result of IS_A1_A2_B2_B1_B1_A2_A1_A2_A1_A2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5586579, upper bound: 0.5421828
time: 0.42 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_A2_A1_A2_A1_A2_A2_A2

### Relational analysis result of IS_A1_A2_B2_B1_B1_A2_A1_A2_A1_A2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5464140, upper bound: 0.5582596
time: 0.45 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B2_A1_A1_A1_A1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0965443, 0.3969810, -0.0054744, 0.7100763, -0.6135321, 0.4024554
1: 0.0565891, 0.4885263, -0.0693125, 0.9573513, -0.9007622, 0.5578388
2: 0.0083284, 0.4923123, -0.1748363, 0.7621485, -0.7538201, 0.6671486
3: -0.0780025, 0.4490798, -0.2408400, 0.9047788, -0.9827812, 0.6899198
4: -0.0766807, 0.5934064, -0.2802821, 1.0476310, -1.1243117, 0.8736885

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B2_B1_B2_A1_A1_A1_A1_A1_A1_A1

### Relational analysis result of IS_A1_A2_B2_B1_B2_A1_A1_A1_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5683143, upper bound: 0.5109562
time: 0.42 seconds

## Relational analysis of IS_A1_A2_B2_B1_B2_A1_A1_A1_A1_A1_A1_A2

### Relational analysis result of IS_A1_A2_B2_B1_B2_A1_A1_A1_A1_A1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5645366, upper bound: 0.5109562
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_B1_B1_B1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0069308, 0.7141405, 0.0965443, 0.3969810, -0.4039118, 0.6175963
1: -0.0711197, 0.9631418, 0.0565891, 0.4885263, -0.5596460, 0.9065527
2: -0.1773483, 0.7663486, 0.0083284, 0.4923123, -0.6696606, 0.7580202
3: -0.2441539, 0.9109904, -0.0780025, 0.4490798, -0.6932336, 0.9889928
4: -0.2841340, 1.0538890, -0.0766807, 0.5934064, -0.8775404, 1.1305697

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_B2_A1_B1_B1_B1_B1_B1_B1_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_B1_B1_B1_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5111559, upper bound: 0.5684190
time: 0.41 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_B1_B1_B1_B1_B1_B1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_B1_B1_B1_B1_B1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5111559, upper bound: 0.5598047
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B1_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0433788, 0.4866467, 0.0873206, 0.4219245, -0.3785456, 0.3993261
1: -0.0069845, 0.5967795, 0.0395762, 0.5401090, -0.5470935, 0.5572033
2: -0.0581915, 0.5744838, -0.0181572, 0.5037278, -0.5619193, 0.5926410
3: -0.1468202, 0.5730036, -0.0989394, 0.4913468, -0.6381670, 0.6719431
4: -0.1378696, 0.6996939, -0.0983677, 0.6463114, -0.7841810, 0.7980616

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_B2_A2_B1_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5415877, upper bound: 0.5654935
time: 0.45 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5304913, upper bound: 0.5654935
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B1_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0283030, 0.5522523, 0.0873206, 0.4219245, -0.3936214, 0.4649317
1: -0.0250288, 0.7141986, 0.0395762, 0.5401090, -0.5651378, 0.6746224
2: -0.0988264, 0.6263493, -0.0181572, 0.5037278, -0.6025542, 0.6445066
3: -0.1721501, 0.6781709, -0.0989394, 0.4913468, -0.6634969, 0.7771103
4: -0.1945045, 0.8138232, -0.0983677, 0.6463114, -0.8408159, 0.9121909

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_B2_A2_B1_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5304913, upper bound: 0.5654935
time: 0.42 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5304913, upper bound: 0.5654935
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B1_A1_B1_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: 0.0522337, 0.4859760, 0.0873206, 0.4219245, -0.3696908, 0.3986554
1: 0.0017436, 0.5758268, 0.0395762, 0.5401090, -0.5383654, 0.5362506
2: -0.0330665, 0.5877808, -0.0181572, 0.5037278, -0.5367943, 0.6059381
3: -0.1326582, 0.5599031, -0.0989394, 0.4913468, -0.6240050, 0.6588426
4: -0.1118598, 0.6914989, -0.0983677, 0.6463114, -0.7581712, 0.7898666

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_B2_A2_B1_A1_B1_A2_B1_A1_A2_A1

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1_B1_A2_B1_A1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5415877, upper bound: 0.5507922
time: 0.45 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_A1_B1_A2_B1_A1_A2_A2

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1_B1_A2_B1_A1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5304913, upper bound: 0.5507922
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B1_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0283030, 0.5522523, 0.0873206, 0.4219245, -0.3936214, 0.4649317
1: -0.0250288, 0.7141986, 0.0395762, 0.5401090, -0.5651378, 0.6746224
2: -0.0988264, 0.6263493, -0.0181572, 0.5037278, -0.6025542, 0.6445066
3: -0.1721501, 0.6781709, -0.0989394, 0.4913468, -0.6634969, 0.7771103
4: -0.1945045, 0.8138232, -0.0983677, 0.6463114, -0.8408159, 0.9121909

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_B2_A2_B1_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5304913, upper bound: 0.5654935
time: 0.41 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5304913, upper bound: 0.5654935
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B1_A2_B1_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: 0.0522337, 0.4859760, 0.0873206, 0.4219245, -0.3696908, 0.3986554
1: 0.0017436, 0.5758268, 0.0395762, 0.5401090, -0.5383654, 0.5362506
2: -0.0330665, 0.5877808, -0.0181572, 0.5037278, -0.5367943, 0.6059381
3: -0.1326582, 0.5599031, -0.0989394, 0.4913468, -0.6240050, 0.6588426
4: -0.1118598, 0.6914989, -0.0983677, 0.6463114, -0.7581712, 0.7898666

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_B2_A2_B1_A2_B1_A2_B1_A1_A2_A1

### Relational analysis result of IS_A2_B1_B2_A2_B1_A2_B1_A2_B1_A1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5415877, upper bound: 0.5507922
time: 0.44 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_A2_B1_A2_B1_A1_A2_A2

### Relational analysis result of IS_A2_B1_B2_A2_B1_A2_B1_A2_B1_A1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5304913, upper bound: 0.5507922
time: 0.44 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B1_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0233569, 0.5570214, 0.0873206, 0.4219245, -0.3985676, 0.4697009
1: -0.0314330, 0.7206672, 0.0395762, 0.5401090, -0.5715420, 0.6810911
2: -0.1071281, 0.6316106, -0.0181572, 0.5037278, -0.6108559, 0.6497678
3: -0.1827013, 0.6866775, -0.0989394, 0.4913468, -0.6740481, 0.7856169
4: -0.2049841, 0.8204701, -0.0983677, 0.6463114, -0.8512955, 0.9188378

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_B2_A2_B1_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_B2_A2_B1_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5304913, upper bound: 0.5654935
time: 0.41 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5304913, upper bound: 0.5754925
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B2_B1_B1_B1_B1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0083361, 0.5719668, 0.0969255, 0.3960601, -0.3877240, 0.4750413
1: -0.0506135, 0.7433118, 0.0570602, 0.4867831, -0.5373967, 0.6862516
2: -0.1297052, 0.6455543, 0.0090892, 0.4917682, -0.6214734, 0.6364651
3: -0.2118729, 0.7126579, -0.0774024, 0.4474990, -0.6593719, 0.7900603
4: -0.2322460, 0.8428500, -0.0759368, 0.5922844, -0.8245305, 0.9187868

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B1_B1_B1_B1_B1_B1

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1_B1_B1_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5797783
time: 0.43 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B1_B1_B1_B1_B1_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1_B1_B1_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5754925
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B2_B1_B1_B1_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0083361, 0.5719668, 0.0925431, 0.3798722, -0.3715361, 0.4794236
1: -0.0506135, 0.7433118, 0.0508025, 0.4456694, -0.4962829, 0.6925094
2: -0.1297052, 0.6455543, 0.0190966, 0.4847844, -0.6144896, 0.6264577
3: -0.2118729, 0.7126579, -0.0695522, 0.4218645, -0.6337374, 0.7822101
4: -0.2322460, 0.8428500, -0.0605929, 0.5501102, -0.7823563, 0.9034429

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B1_B1_B1_B1_B2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1_B1_B1_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5724233
time: 0.45 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B1_B1_B1_B1_B2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1_B1_B1_B1_B1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5611240
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B2_B1_B1_B1_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0274241, 0.5315738, 0.0937147, 0.4119777, -0.3845536, 0.4378591
1: -0.0291177, 0.6686592, 0.0489067, 0.5234078, -0.5525255, 0.6197525
2: -0.0887789, 0.6183543, -0.0067511, 0.4970919, -0.5858707, 0.6251054
3: -0.1754975, 0.6463004, -0.0869917, 0.4748205, -0.6503180, 0.7332922
4: -0.1838146, 0.7758477, -0.0889151, 0.6335514, -0.8173660, 0.8647628

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B1_B1_B1_B2_A1_A1

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1_B1_B1_B1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5464466
time: 0.43 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B1_B1_B1_B2_A1_A2

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1_B1_B1_B1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5608028
time: 0.44 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B2_B1_B1_B1_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0149566, 0.5658405, 0.0937147, 0.4119777, -0.3970211, 0.4721258
1: -0.0463459, 0.7114239, 0.0489067, 0.5234078, -0.5697538, 0.6625172
2: -0.1064991, 0.6583920, -0.0067511, 0.4970919, -0.6035910, 0.6651430
3: -0.2061497, 0.6969915, -0.0869917, 0.4748205, -0.6809702, 0.7839832
4: -0.2113429, 0.8311484, -0.0889151, 0.6335514, -0.8448943, 0.9200635

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B1_B1_B1_B2_A2_A1

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1_B1_B1_B1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5507922
time: 0.49 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B1_B1_B1_B2_A2_A2

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1_B1_B1_B1_B2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5611240
time: 0.44 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B2_B1_B1_B1_B2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0083361, 0.5719668, 0.0969255, 0.3960601, -0.3877240, 0.4750413
1: -0.0506135, 0.7433118, 0.0570602, 0.4867831, -0.5373967, 0.6862516
2: -0.1297052, 0.6455543, 0.0090892, 0.4917682, -0.6214734, 0.6364651
3: -0.2118729, 0.7126579, -0.0774024, 0.4474990, -0.6593719, 0.7900603
4: -0.2322460, 0.8428500, -0.0759368, 0.5922844, -0.8245305, 0.9187868

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B1_B1_B2_B1_B1_B1

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1_B1_B1_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5797783
time: 0.44 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B1_B1_B2_B1_B1_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1_B1_B1_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5754925
time: 0.44 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B2_B1_B1_B1_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0083361, 0.5719668, 0.0925431, 0.3798722, -0.3715361, 0.4794236
1: -0.0506135, 0.7433118, 0.0508025, 0.4456694, -0.4962829, 0.6925094
2: -0.1297052, 0.6455543, 0.0190966, 0.4847844, -0.6144896, 0.6264577
3: -0.2118729, 0.7126579, -0.0695522, 0.4218645, -0.6337374, 0.7822101
4: -0.2322460, 0.8428500, -0.0605929, 0.5501102, -0.7823563, 0.9034429

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B1_B1_B2_B1_B2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1_B1_B1_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5724233
time: 0.46 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B1_B1_B2_B1_B2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1_B1_B1_B2_B1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5611240
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B2_B1_B2_B1_B1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0083361, 0.5719668, 0.0969255, 0.3960601, -0.3877240, 0.4750413
1: -0.0506135, 0.7433118, 0.0570602, 0.4867831, -0.5373967, 0.6862516
2: -0.1297052, 0.6455543, 0.0090892, 0.4917682, -0.6214734, 0.6364651
3: -0.2118729, 0.7126579, -0.0774024, 0.4474990, -0.6593719, 0.7900603
4: -0.2322460, 0.8428500, -0.0759368, 0.5922844, -0.8245305, 0.9187868

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B2_B1_B1_B1_B1_B1

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1_B2_B1_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5797783
time: 0.43 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B2_B1_B1_B1_B1_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1_B2_B1_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5754925
time: 0.45 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B2_B1_B2_B1_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0083361, 0.5719668, 0.0925431, 0.3798722, -0.3715361, 0.4794236
1: -0.0506135, 0.7433118, 0.0508025, 0.4456694, -0.4962829, 0.6925094
2: -0.1297052, 0.6455543, 0.0190966, 0.4847844, -0.6144896, 0.6264577
3: -0.2118729, 0.7126579, -0.0695522, 0.4218645, -0.6337374, 0.7822101
4: -0.2322460, 0.8428500, -0.0605929, 0.5501102, -0.7823563, 0.9034429

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B2_B1_B1_B1_B2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1_B2_B1_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5724233
time: 0.46 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B2_B1_B1_B1_B2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1_B2_B1_B1_B1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5611240
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B2_B1_B2_B1_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0274241, 0.5315738, 0.0937147, 0.4119777, -0.3845536, 0.4378591
1: -0.0291177, 0.6686592, 0.0489067, 0.5234078, -0.5525255, 0.6197525
2: -0.0887789, 0.6183543, -0.0067511, 0.4970919, -0.5858707, 0.6251054
3: -0.1754975, 0.6463004, -0.0869917, 0.4748205, -0.6503180, 0.7332922
4: -0.1838146, 0.7758477, -0.0889151, 0.6335514, -0.8173660, 0.8647628

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B2_B1_B1_B2_A1_A1

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1_B2_B1_B1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5464466
time: 0.44 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B2_B1_B1_B2_A1_A2

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1_B2_B1_B1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5608028
time: 0.44 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B2_B1_B2_B1_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0149566, 0.5658405, 0.0937147, 0.4119777, -0.3970211, 0.4721258
1: -0.0463459, 0.7114239, 0.0489067, 0.5234078, -0.5697538, 0.6625172
2: -0.1064991, 0.6583920, -0.0067511, 0.4970919, -0.6035910, 0.6651430
3: -0.2061497, 0.6969915, -0.0869917, 0.4748205, -0.6809702, 0.7839832
4: -0.2113429, 0.8311484, -0.0889151, 0.6335514, -0.8448943, 0.9200635

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B2_B1_B1_B2_A2_A1

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1_B2_B1_B1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5507922
time: 0.44 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B2_B1_B1_B2_A2_A2

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1_B2_B1_B1_B2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5611240
time: 0.45 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B2_B1_B2_B1_B2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0083361, 0.5719668, 0.0969255, 0.3960601, -0.3877240, 0.4750413
1: -0.0506135, 0.7433118, 0.0570602, 0.4867831, -0.5373967, 0.6862516
2: -0.1297052, 0.6455543, 0.0090892, 0.4917682, -0.6214734, 0.6364651
3: -0.2118729, 0.7126579, -0.0774024, 0.4474990, -0.6593719, 0.7900603
4: -0.2322460, 0.8428500, -0.0759368, 0.5922844, -0.8245305, 0.9187868

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B2_B1_B2_B1_B1_B1

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1_B2_B1_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5797783
time: 0.43 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B2_B1_B2_B1_B1_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1_B2_B1_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5754925
time: 0.45 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B2_B1_B2_B1_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0083361, 0.5719668, 0.0921624, 0.3807045, -0.3723685, 0.4798044
1: -0.0506135, 0.7433118, 0.0501726, 0.4466347, -0.4972482, 0.6931393
2: -0.1297052, 0.6455543, 0.0185485, 0.4862787, -0.6159838, 0.6270058
3: -0.2118729, 0.7126579, -0.0701120, 0.4227494, -0.6346223, 0.7827699
4: -0.2322460, 0.8428500, -0.0608306, 0.5514524, -0.7836984, 0.9036806

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B2_B1_B2_B1_B2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1_B2_B1_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5197859, upper bound: 0.5724233
time: 0.41 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B2_B1_B2_B1_B2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1_B2_B1_B2_B1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5197859, upper bound: 0.5611240
time: 0.42 seconds

## Summary of splitting at layer (split count: 11)
- Time for IS candidates: 2.91 seconds
IS_A1_A2_B2_B1_B1_A1_A1_A1_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 12, time: 2.91
Output dim: 0, lower bound: -0.5696524, upper bound: 0.4702711
IS_A1_A2_B2_B1_B1_A1_A1_A1_A1_A1_A1_A2, status: Status.UNKNOWN, split count: 12, time: 2.91
Output dim: 0, lower bound: -0.5694090, upper bound: 0.4702711
IS_A1_A2_B2_B1_B1_A1_A1_A1_A1_A2_A1_A1, status: Status.UNKNOWN, split count: 12, time: 2.91
Output dim: 0, lower bound: -0.5694090, upper bound: 0.4702711
IS_A1_A2_B2_B1_B1_A1_A1_A1_A1_A2_A1_A2, status: Status.UNKNOWN, split count: 12, time: 2.91
Output dim: 0, lower bound: -0.5694090, upper bound: 0.4702711
IS_A1_A2_B2_B1_B1_A1_A1_A2_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 12, time: 2.91
Output dim: 0, lower bound: -0.5696524, upper bound: 0.4702711
IS_A1_A2_B2_B1_B1_A1_A1_A2_A1_A1_A1_A2, status: Status.UNKNOWN, split count: 12, time: 2.91
Output dim: 0, lower bound: -0.5694090, upper bound: 0.4702711
IS_A1_A2_B2_B1_B1_A1_A1_A2_A1_A2_A1_A1, status: Status.UNKNOWN, split count: 12, time: 2.91
Output dim: 0, lower bound: -0.5694090, upper bound: 0.4702711
IS_A1_A2_B2_B1_B1_A1_A1_A2_A1_A2_A1_A2, status: Status.UNKNOWN, split count: 12, time: 2.91
Output dim: 0, lower bound: -0.5694090, upper bound: 0.4702711
IS_A1_A2_B2_B1_B1_A2_A1_A1_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 12, time: 2.91
Output dim: 0, lower bound: -0.5663972, upper bound: 0.4605393
IS_A1_A2_B2_B1_B1_A2_A1_A1_A1_A1_A1_A2, status: Status.UNKNOWN, split count: 12, time: 2.91
Output dim: 0, lower bound: -0.5662008, upper bound: 0.4605393
IS_A1_A2_B2_B1_B1_A2_A1_A1_A1_A2_A1_A1, status: Status.UNKNOWN, split count: 12, time: 2.91
Output dim: 0, lower bound: -0.5662008, upper bound: 0.4605393
IS_A1_A2_B2_B1_B1_A2_A1_A1_A1_A2_A1_A2, status: Status.UNKNOWN, split count: 12, time: 2.91
Output dim: 0, lower bound: -0.5662008, upper bound: 0.4605393
IS_A1_A2_B2_B1_B1_A2_A1_A2_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 12, time: 2.91
Output dim: 0, lower bound: -0.5663972, upper bound: 0.4605393
IS_A1_A2_B2_B1_B1_A2_A1_A2_A1_A1_A1_A2, status: Status.UNKNOWN, split count: 12, time: 2.91
Output dim: 0, lower bound: -0.5662008, upper bound: 0.4605393
IS_A1_A2_B2_B1_B1_A2_A1_A2_A1_A2_A1_A1, status: Status.UNKNOWN, split count: 12, time: 2.91
Output dim: 0, lower bound: -0.5674961, upper bound: 0.4893037
IS_A1_A2_B2_B1_B1_A2_A1_A2_A1_A2_A1_A2, status: Status.UNKNOWN, split count: 12, time: 2.91
Output dim: 0, lower bound: -0.5674961, upper bound: 0.5696298
IS_A1_A2_B2_B1_B1_A2_A1_A2_A1_A2_A2_A1, status: Status.VERIFIED, split count: 12, time: 2.91
Output dim: 0, lower bound: -0.5586579, upper bound: 0.5421828
IS_A1_A2_B2_B1_B1_A2_A1_A2_A1_A2_A2_A2, status: Status.VERIFIED, split count: 12, time: 2.91
Output dim: 0, lower bound: -0.5464140, upper bound: 0.5582596
IS_A1_A2_B2_B1_B2_A1_A1_A1_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 12, time: 2.91
Output dim: 0, lower bound: -0.5683143, upper bound: 0.5109562
IS_A1_A2_B2_B1_B2_A1_A1_A1_A1_A1_A1_A2, status: Status.VERIFIED, split count: 12, time: 2.91
Output dim: 0, lower bound: -0.5645366, upper bound: 0.5109562
IS_A2_B1_B2_A1_B1_B1_B1_B1_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 12, time: 2.91
Output dim: 0, lower bound: -0.5111559, upper bound: 0.5684190
IS_A2_B1_B2_A1_B1_B1_B1_B1_B1_B1_B1_B2, status: Status.VERIFIED, split count: 12, time: 2.91
Output dim: 0, lower bound: -0.5111559, upper bound: 0.5598047
IS_A2_B1_B2_A2_B1_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 12, time: 2.91
Output dim: 0, lower bound: -0.5415877, upper bound: 0.5654935
IS_A2_B1_B2_A2_B1_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 12, time: 2.91
Output dim: 0, lower bound: -0.5304913, upper bound: 0.5654935
IS_A2_B1_B2_A2_B1_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 12, time: 2.91
Output dim: 0, lower bound: -0.5304913, upper bound: 0.5654935
IS_A2_B1_B2_A2_B1_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 12, time: 2.91
Output dim: 0, lower bound: -0.5304913, upper bound: 0.5654935
IS_A2_B1_B2_A2_B1_A1_B1_A2_B1_A1_A2_A1, status: Status.VERIFIED, split count: 12, time: 2.91
Output dim: 0, lower bound: -0.5415877, upper bound: 0.5507922
IS_A2_B1_B2_A2_B1_A1_B1_A2_B1_A1_A2_A2, status: Status.VERIFIED, split count: 12, time: 2.91
Output dim: 0, lower bound: -0.5304913, upper bound: 0.5507922
IS_A2_B1_B2_A2_B1_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 12, time: 2.91
Output dim: 0, lower bound: -0.5304913, upper bound: 0.5654935
IS_A2_B1_B2_A2_B1_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 12, time: 2.91
Output dim: 0, lower bound: -0.5304913, upper bound: 0.5654935
IS_A2_B1_B2_A2_B1_A2_B1_A2_B1_A1_A2_A1, status: Status.VERIFIED, split count: 12, time: 2.91
Output dim: 0, lower bound: -0.5415877, upper bound: 0.5507922
IS_A2_B1_B2_A2_B1_A2_B1_A2_B1_A1_A2_A2, status: Status.VERIFIED, split count: 12, time: 2.91
Output dim: 0, lower bound: -0.5304913, upper bound: 0.5507922
IS_A2_B1_B2_A2_B1_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 12, time: 2.91
Output dim: 0, lower bound: -0.5304913, upper bound: 0.5654935
IS_A2_B1_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 12, time: 2.91
Output dim: 0, lower bound: -0.5304913, upper bound: 0.5754925
IS_A2_B1_B2_A2_B2_B1_B1_B1_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 12, time: 2.91
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5797783
IS_A2_B1_B2_A2_B2_B1_B1_B1_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 12, time: 2.91
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5754925
IS_A2_B1_B2_A2_B2_B1_B1_B1_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 12, time: 2.91
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5724233
IS_A2_B1_B2_A2_B2_B1_B1_B1_B1_B1_B2_B2, status: Status.VERIFIED, split count: 12, time: 2.91
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5611240
IS_A2_B1_B2_A2_B2_B1_B1_B1_B1_B2_A1_A1, status: Status.VERIFIED, split count: 12, time: 2.91
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5464466
IS_A2_B1_B2_A2_B2_B1_B1_B1_B1_B2_A1_A2, status: Status.VERIFIED, split count: 12, time: 2.91
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5608028
IS_A2_B1_B2_A2_B2_B1_B1_B1_B1_B2_A2_A1, status: Status.VERIFIED, split count: 12, time: 2.91
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5507922
IS_A2_B1_B2_A2_B2_B1_B1_B1_B1_B2_A2_A2, status: Status.VERIFIED, split count: 12, time: 2.91
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5611240
IS_A2_B1_B2_A2_B2_B1_B1_B1_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 12, time: 2.91
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5797783
IS_A2_B1_B2_A2_B2_B1_B1_B1_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 12, time: 2.91
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5754925
IS_A2_B1_B2_A2_B2_B1_B1_B1_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 12, time: 2.91
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5724233
IS_A2_B1_B2_A2_B2_B1_B1_B1_B2_B1_B2_B2, status: Status.VERIFIED, split count: 12, time: 2.91
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5611240
IS_A2_B1_B2_A2_B2_B1_B2_B1_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 12, time: 2.91
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5797783
IS_A2_B1_B2_A2_B2_B1_B2_B1_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 12, time: 2.91
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5754925
IS_A2_B1_B2_A2_B2_B1_B2_B1_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 12, time: 2.91
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5724233
IS_A2_B1_B2_A2_B2_B1_B2_B1_B1_B1_B2_B2, status: Status.VERIFIED, split count: 12, time: 2.91
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5611240
IS_A2_B1_B2_A2_B2_B1_B2_B1_B1_B2_A1_A1, status: Status.VERIFIED, split count: 12, time: 2.91
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5464466
IS_A2_B1_B2_A2_B2_B1_B2_B1_B1_B2_A1_A2, status: Status.VERIFIED, split count: 12, time: 2.91
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5608028
IS_A2_B1_B2_A2_B2_B1_B2_B1_B1_B2_A2_A1, status: Status.VERIFIED, split count: 12, time: 2.91
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5507922
IS_A2_B1_B2_A2_B2_B1_B2_B1_B1_B2_A2_A2, status: Status.VERIFIED, split count: 12, time: 2.91
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5611240
IS_A2_B1_B2_A2_B2_B1_B2_B1_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 12, time: 2.91
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5797783
IS_A2_B1_B2_A2_B2_B1_B2_B1_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 12, time: 2.91
Output dim: 0, lower bound: -0.5144045, upper bound: 0.5754925
IS_A2_B1_B2_A2_B2_B1_B2_B1_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 12, time: 2.91
Output dim: 0, lower bound: -0.5197859, upper bound: 0.5724233
IS_A2_B1_B2_A2_B2_B1_B2_B1_B2_B1_B2_B2, status: Status.VERIFIED, split count: 12, time: 2.91
Output dim: 0, lower bound: -0.5197859, upper bound: 0.5611240

## BFS IS instance: IS_A1_A2_B2_B1_B1_A1_A1_A1_A1_A1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0945764, 0.3505142, 0.0646830, 0.5288626, -0.4342862, 0.2845256
1: 0.0685086, 0.3967934, 0.0195227, 0.7196292, -0.6511206, 0.3772707
2: 0.0585651, 0.4334365, -0.0684017, 0.5826666, -0.5241014, 0.5018382
3: -0.0406969, 0.4090275, -0.1308922, 0.6449826, -0.6856795, 0.5399196
4: -0.0057726, 0.4402502, -0.1473095, 0.8047376, -0.8105102, 0.5875597

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B2_B1_B1_A1_A1_A1_A1_A1_A1_A1_A1

### Relational analysis result of IS_A1_A2_B2_B1_B1_A1_A1_A1_A1_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5663972, upper bound: 0.4605393
time: 0.45 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_A1_A1_A1_A1_A1_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B2_B1_B1_A1_A1_A1_A1_A1_A1_A1_A1

### Relational analysis result of IS_A1_A2_B2_B1_B1_A1_A1_A1_A1_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5696524, upper bound: 0.4702711
time: 0.43 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_A1_A1_A1_A1_A1_A1_A1_A2

### Relational analysis result of IS_A1_A2_B2_B1_B1_A1_A1_A1_A1_A1_A1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5617467, upper bound: 0.4702711
time: 0.43 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B1_A1_A1_A1_A1_A1_A1_A2

### Backsubstitution after applying IS history:
0: 0.0924387, 0.3566364, 0.0644008, 0.5302621, -0.4378234, 0.2922356
1: 0.0503964, 0.4023641, 0.0192177, 0.7222201, -0.6718237, 0.3831464
2: 0.0225284, 0.5009353, -0.0694884, 0.5835138, -0.5609854, 0.5704237
3: -0.0864439, 0.3917682, -0.1317451, 0.6470909, -0.7335348, 0.5235133
4: -0.0891201, 0.5688827, -0.1483382, 0.8069319, -0.8960520, 0.7172209

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B2_B1_B1_A1_A1_A1_A1_A1_A1_A2_A1

### Relational analysis result of IS_A1_A2_B2_B1_B1_A1_A1_A1_A1_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5694090, upper bound: 0.4702711
time: 0.42 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_A1_A1_A1_A1_A1_A1_A2_A2

### Relational analysis result of IS_A1_A2_B2_B1_B1_A1_A1_A1_A1_A1_A1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5613755, upper bound: 0.4702711
time: 0.42 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B1_A1_A1_A1_A1_A2_A1_A1

### Backsubstitution after applying IS history:
0: 0.0947785, 0.3503857, 0.0646830, 0.5288626, -0.4340841, 0.2844018
1: 0.0685635, 0.3965145, 0.0195227, 0.7196292, -0.6510658, 0.3769919
2: 0.0587087, 0.4331901, -0.0684017, 0.5826666, -0.5239579, 0.5015918
3: -0.0405765, 0.4087051, -0.1308922, 0.6449826, -0.6855591, 0.5395973
4: -0.0056551, 0.4400306, -0.1473095, 0.8047376, -0.8103926, 0.5873401

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B2_B1_B1_A1_A1_A1_A1_A2_A1_A1_A1

### Relational analysis result of IS_A1_A2_B2_B1_B1_A1_A1_A1_A1_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5663972, upper bound: 0.4605393
time: 0.45 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_A1_A1_A1_A1_A2_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B2_B1_B1_A1_A1_A1_A1_A2_A1_A1_A1

### Relational analysis result of IS_A1_A2_B2_B1_B1_A1_A1_A1_A1_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5696524, upper bound: 0.4702711
time: 0.43 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_A1_A1_A1_A1_A2_A1_A1_A2

### Relational analysis result of IS_A1_A2_B2_B1_B1_A1_A1_A1_A1_A2_A1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5617467, upper bound: 0.4702711
time: 0.42 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B1_A1_A1_A1_A1_A2_A1_A2

### Backsubstitution after applying IS history:
0: 0.0924387, 0.3566364, 0.0644008, 0.5302621, -0.4378234, 0.2922356
1: 0.0503964, 0.4023641, 0.0192177, 0.7222201, -0.6718237, 0.3831464
2: 0.0225284, 0.5009353, -0.0694884, 0.5835138, -0.5609854, 0.5704237
3: -0.0864439, 0.3917682, -0.1317451, 0.6470909, -0.7335348, 0.5235133
4: -0.0891201, 0.5688827, -0.1483382, 0.8069319, -0.8960520, 0.7172209

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B2_B1_B1_A1_A1_A1_A1_A2_A1_A2_A1

### Relational analysis result of IS_A1_A2_B2_B1_B1_A1_A1_A1_A1_A2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5694090, upper bound: 0.4702711
time: 0.41 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_A1_A1_A1_A1_A2_A1_A2_A2

### Relational analysis result of IS_A1_A2_B2_B1_B1_A1_A1_A1_A1_A2_A1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5613755, upper bound: 0.4702711
time: 0.43 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B1_A1_A1_A2_A1_A1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0947785, 0.3503857, 0.0646830, 0.5288626, -0.4340841, 0.2844018
1: 0.0685635, 0.3965145, 0.0195227, 0.7196292, -0.6510658, 0.3769919
2: 0.0587087, 0.4331901, -0.0684017, 0.5826666, -0.5239579, 0.5015918
3: -0.0405765, 0.4087051, -0.1308922, 0.6449826, -0.6855591, 0.5395973
4: -0.0056551, 0.4400306, -0.1473095, 0.8047376, -0.8103926, 0.5873401

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B2_B1_B1_A1_A1_A2_A1_A1_A1_A1_A1

### Relational analysis result of IS_A1_A2_B2_B1_B1_A1_A1_A2_A1_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5663972, upper bound: 0.4605393
time: 0.45 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_A1_A1_A2_A1_A1_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B2_B1_B1_A1_A1_A2_A1_A1_A1_A1_A1

### Relational analysis result of IS_A1_A2_B2_B1_B1_A1_A1_A2_A1_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5696524, upper bound: 0.4702711
time: 0.43 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_A1_A1_A2_A1_A1_A1_A1_A2

### Relational analysis result of IS_A1_A2_B2_B1_B1_A1_A1_A2_A1_A1_A1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5617467, upper bound: 0.4702711
time: 0.42 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B1_A1_A1_A2_A1_A1_A1_A2

### Backsubstitution after applying IS history:
0: 0.0924387, 0.3566364, 0.0644008, 0.5302621, -0.4378234, 0.2922356
1: 0.0503964, 0.4023641, 0.0192177, 0.7222201, -0.6718237, 0.3831464
2: 0.0225284, 0.5009353, -0.0694884, 0.5835138, -0.5609854, 0.5704237
3: -0.0864439, 0.3917682, -0.1317451, 0.6470909, -0.7335348, 0.5235133
4: -0.0891201, 0.5688827, -0.1483382, 0.8069319, -0.8960520, 0.7172209

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B2_B1_B1_A1_A1_A2_A1_A1_A1_A2_A1

### Relational analysis result of IS_A1_A2_B2_B1_B1_A1_A1_A2_A1_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5694090, upper bound: 0.4702711
time: 0.42 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_A1_A1_A2_A1_A1_A1_A2_A2

### Relational analysis result of IS_A1_A2_B2_B1_B1_A1_A1_A2_A1_A1_A1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5613755, upper bound: 0.4702711
time: 0.43 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B1_A1_A1_A2_A1_A2_A1_A1

### Backsubstitution after applying IS history:
0: 0.0947785, 0.3503857, 0.0646830, 0.5288626, -0.4340841, 0.2844018
1: 0.0685635, 0.3965145, 0.0195227, 0.7196292, -0.6510658, 0.3769919
2: 0.0587087, 0.4331901, -0.0684017, 0.5826666, -0.5239579, 0.5015918
3: -0.0405765, 0.4087051, -0.1308922, 0.6449826, -0.6855591, 0.5395973
4: -0.0056551, 0.4400306, -0.1473095, 0.8047376, -0.8103926, 0.5873401

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B2_B1_B1_A1_A1_A2_A1_A2_A1_A1_A1

### Relational analysis result of IS_A1_A2_B2_B1_B1_A1_A1_A2_A1_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5663972, upper bound: 0.4605393
time: 0.44 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_A1_A1_A2_A1_A2_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B2_B1_B1_A1_A1_A2_A1_A2_A1_A1_A1

### Relational analysis result of IS_A1_A2_B2_B1_B1_A1_A1_A2_A1_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5696524, upper bound: 0.4702711
time: 0.43 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_A1_A1_A2_A1_A2_A1_A1_A2

### Relational analysis result of IS_A1_A2_B2_B1_B1_A1_A1_A2_A1_A2_A1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5617467, upper bound: 0.4702711
time: 0.43 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B1_A1_A1_A2_A1_A2_A1_A2

### Backsubstitution after applying IS history:
0: 0.0924387, 0.3566364, 0.0644008, 0.5302621, -0.4378234, 0.2922356
1: 0.0503964, 0.4023641, 0.0192177, 0.7222201, -0.6718237, 0.3831464
2: 0.0225284, 0.5009353, -0.0694884, 0.5835138, -0.5609854, 0.5704237
3: -0.0864439, 0.3917682, -0.1317451, 0.6470909, -0.7335348, 0.5235133
4: -0.0891201, 0.5688827, -0.1483382, 0.8069319, -0.8960520, 0.7172209

Time for backsubstitution: 1.95 seconds
Binary search (step 1): status=Status.UNKNOWN, low=0.0170836, high=0.0582672, mid=0.0582672, abs_max=0.6789970397949219
rel_dist={0: [-0.5863374219570315, 0.5863374219570305]}

## Binary search (step 2) starts
Candidate diff: 0.0376754


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5758081, upper bound: 0.5559249
time: 0.33 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5758081, upper bound: 0.5758081
time: 0.35 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.83 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.83
Output dim: 0, lower bound: -0.5758081, upper bound: 0.5559249
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.83
Output dim: 0, lower bound: -0.5758081, upper bound: 0.5758081

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.0674889, 0.4535027, -0.0197046, 0.5769862, -0.5094973, 0.4732073
1: 0.0201817, 0.5976362, -0.0782405, 0.7587138, -0.7385322, 0.6758767
2: -0.0460193, 0.5242411, -0.1620442, 0.6439690, -0.6899883, 0.6862853
3: -0.1192396, 0.5392005, -0.2490228, 0.7361293, -0.8553689, 0.7882234
4: -0.1227688, 0.6923177, -0.2728895, 0.8465014, -0.9692701, 0.9652072

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5684869, upper bound: 0.5533205
time: 0.35 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5739817, upper bound: 0.5541649
time: 0.38 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0106674, 0.5881721, -0.0306828, 0.6307013, -0.6413687, 0.6188549
1: -0.0680232, 0.7645921, -0.0919231, 0.8260971, -0.8941203, 0.8565152
2: -0.1492940, 0.6635130, -0.1838701, 0.7013786, -0.8506727, 0.8473831
3: -0.2353030, 0.7381678, -0.2745254, 0.8015700, -1.0368731, 1.0126932
4: -0.2566378, 0.8659467, -0.3016382, 0.9287875, -1.1854253, 1.1675849

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5559249, upper bound: 0.5758081
time: 0.35 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5559249, upper bound: 0.5758081
time: 0.36 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.35 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 2.35
Output dim: 0, lower bound: -0.5684869, upper bound: 0.5533205
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 2.35
Output dim: 0, lower bound: -0.5739817, upper bound: 0.5541649
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.35
Output dim: 0, lower bound: -0.5559249, upper bound: 0.5758081
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.35
Output dim: 0, lower bound: -0.5559249, upper bound: 0.5758081

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: 0.0667267, 0.5276489, -0.0144424, 0.5661175, -0.4993908, 0.5420914
1: 0.0209925, 0.7202063, -0.0717734, 0.7437310, -0.7227385, 0.7919797
2: -0.0676523, 0.5800259, -0.1531745, 0.6335558, -0.7012081, 0.7332004
3: -0.1276886, 0.6454976, -0.2392942, 0.7200592, -0.8477479, 0.8847918
4: -0.1461909, 0.8046894, -0.2614821, 0.8295491, -0.9757400, 1.0661715

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5647945, upper bound: 0.5450628
time: 0.36 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5647945, upper bound: 0.5533205
time: 0.33 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: 0.0789177, 0.4454548, -0.0197046, 0.5769862, -0.4980685, 0.4651594
1: 0.0311321, 0.5851125, -0.0782405, 0.7587138, -0.7275817, 0.6633530
2: -0.0367235, 0.5172547, -0.1620442, 0.6439690, -0.6806925, 0.6792989
3: -0.1111584, 0.5267107, -0.2490228, 0.7361293, -0.8472877, 0.7757335
4: -0.1149590, 0.6824017, -0.2728895, 0.8465014, -0.9614604, 0.9552912

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5720554, upper bound: 0.5148266
time: 0.37 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5738851, upper bound: 0.5540135
time: 0.39 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0106674, 0.5881721, 0.0674889, 0.4535027, -0.4641702, 0.5206832
1: -0.0680232, 0.7645921, 0.0201817, 0.5976362, -0.6656594, 0.7444105
2: -0.1492940, 0.6635130, -0.0460193, 0.5242411, -0.6735351, 0.7095323
3: -0.2353030, 0.7381678, -0.1192396, 0.5392005, -0.7745036, 0.8574073
4: -0.2566378, 0.8659467, -0.1227688, 0.6923177, -0.9489555, 0.9887154

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5533205, upper bound: 0.5603068
time: 0.36 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5541649, upper bound: 0.5739816
time: 0.35 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0106674, 0.5881721, -0.0106674, 0.5881721, -0.5988396, 0.5988396
1: -0.0680232, 0.7645921, -0.0680232, 0.7645921, -0.8326153, 0.8326153
2: -0.1492940, 0.6635130, -0.1492940, 0.6635130, -0.8128070, 0.8128070
3: -0.2353030, 0.7381678, -0.2353030, 0.7381678, -0.9734708, 0.9734708
4: -0.2566378, 0.8659467, -0.2566378, 0.8659467, -1.1225845, 1.1225845

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5533205, upper bound: 0.5603068
time: 0.34 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5541649, upper bound: 0.5739817
time: 0.37 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.36 seconds
IS_A1_A1_B1, status: Status.VERIFIED, split count: 3, time: 2.36
Output dim: 0, lower bound: -0.5647945, upper bound: 0.5450628
IS_A1_A1_B2, status: Status.VERIFIED, split count: 3, time: 2.36
Output dim: 0, lower bound: -0.5647945, upper bound: 0.5533205
IS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 2.36
Output dim: 0, lower bound: -0.5720554, upper bound: 0.5148266
IS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 2.36
Output dim: 0, lower bound: -0.5738851, upper bound: 0.5540135
IS_A2_B1_B1, status: Status.VERIFIED, split count: 3, time: 2.36
Output dim: 0, lower bound: -0.5533205, upper bound: 0.5603068
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 2.36
Output dim: 0, lower bound: -0.5541649, upper bound: 0.5739816
IS_A2_B2_B1, status: Status.VERIFIED, split count: 3, time: 2.36
Output dim: 0, lower bound: -0.5533205, upper bound: 0.5603068
IS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 2.36
Output dim: 0, lower bound: -0.5541649, upper bound: 0.5739817

## BFS IS instance: IS_A1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0803895, 0.4433728, -0.0417449, 0.6100088, -0.5296193, 0.4851178
1: 0.0330061, 0.5817648, -0.1008809, 0.8426676, -0.8096615, 0.6826456
2: -0.0344943, 0.5156111, -0.2217753, 0.6487585, -0.6832528, 0.7373863
3: -0.1087453, 0.5233867, -0.2981970, 0.8164562, -0.9252015, 0.8215837
4: -0.1128467, 0.6797588, -0.3499995, 0.9156853, -1.0285320, 1.0297583

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24

Time for candidate selection: 4.50 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B1

### Relational analysis result of IS_A1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.5130722
time: 0.37 seconds

## Relational analysis of IS_A1_A2_B1_B2

### Relational analysis result of IS_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635
time: 0.36 seconds

## BFS IS instance: IS_A1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0789177, 0.4454548, -0.0186048, 0.5732471, -0.4943293, 0.4640596
1: 0.0311321, 0.5851125, -0.0768588, 0.7538072, -0.7226751, 0.6619713
2: -0.0367235, 0.5172547, -0.1599643, 0.6401152, -0.6768387, 0.6772190
3: -0.1111584, 0.5267107, -0.2466396, 0.7310796, -0.8422380, 0.7733503
4: -0.1149590, 0.6824017, -0.2699898, 0.8404822, -0.9554412, 0.9523915

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A2_B2_B1

### Relational analysis result of IS_A1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5652368, upper bound: 0.5449055
time: 0.37 seconds

## Relational analysis of IS_A1_A2_B2_B2

### Relational analysis result of IS_A1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5652369, upper bound: 0.5451149
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0106674, 0.5881721, 0.0789177, 0.4454548, -0.4561223, 0.5092544
1: -0.0680232, 0.7645921, 0.0311321, 0.5851125, -0.6531357, 0.7334600
2: -0.1492940, 0.6635130, -0.0367235, 0.5172547, -0.6665487, 0.7002365
3: -0.2353030, 0.7381678, -0.1111584, 0.5267107, -0.7620137, 0.8493261
4: -0.2566378, 0.8659467, -0.1149590, 0.6824017, -0.9390395, 0.9809057

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5450628, upper bound: 0.5653475
time: 0.37 seconds

## Relational analysis of IS_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5450628, upper bound: 0.5653475
time: 0.36 seconds

## BFS IS instance: IS_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0106674, 0.5881721, 0.0083361, 0.5719668, -0.5826342, 0.5798361
1: -0.0680232, 0.7645921, -0.0506135, 0.7433118, -0.8113350, 0.8152056
2: -0.1492940, 0.6635130, -0.1297052, 0.6455543, -0.7948483, 0.7932182
3: -0.2353030, 0.7381678, -0.2118729, 0.7126579, -0.9479610, 0.9500406
4: -0.2566378, 0.8659467, -0.2322460, 0.8428500, -1.0994878, 1.0981927

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_B2_A1

### Relational analysis result of IS_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5450628, upper bound: 0.5614455
time: 0.37 seconds

## Relational analysis of IS_A2_B2_B2_A2

### Relational analysis result of IS_A2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5458074, upper bound: 0.5614455
time: 0.40 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.46 seconds
IS_A1_A2_B1_B1, status: Status.VERIFIED, split count: 4, time: 2.46
Output dim: 0, lower bound: -0.5494671, upper bound: 0.5130722
IS_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 2.46
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635
IS_A1_A2_B2_B1, status: Status.VERIFIED, split count: 4, time: 2.46
Output dim: 0, lower bound: -0.5652368, upper bound: 0.5449055
IS_A1_A2_B2_B2, status: Status.VERIFIED, split count: 4, time: 2.46
Output dim: 0, lower bound: -0.5652369, upper bound: 0.5451149
IS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.46
Output dim: 0, lower bound: -0.5450628, upper bound: 0.5653475
IS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.46
Output dim: 0, lower bound: -0.5450628, upper bound: 0.5653475
IS_A2_B2_B2_A1, status: Status.VERIFIED, split count: 4, time: 2.46
Output dim: 0, lower bound: -0.5450628, upper bound: 0.5614455
IS_A2_B2_B2_A2, status: Status.VERIFIED, split count: 4, time: 2.46
Output dim: 0, lower bound: -0.5458074, upper bound: 0.5614455

## BFS IS instance: IS_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0803895, 0.4433728, -0.0208821, 0.5835277, -0.5031382, 0.4642550
1: 0.0330061, 0.5817648, -0.0742011, 0.8010436, -0.7680375, 0.6559659
2: -0.0344943, 0.5156111, -0.1870105, 0.6244919, -0.6589862, 0.7026216
3: -0.1087453, 0.5233867, -0.2587125, 0.7709957, -0.8797410, 0.7820991
4: -0.1128467, 0.6797588, -0.3087695, 0.8734686, -0.9863154, 0.9885283

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_B1

### Relational analysis result of IS_A1_A2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5638812, upper bound: 0.5060034
time: 0.36 seconds

## Relational analysis of IS_A1_A2_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A1

### Relational analysis result of IS_A1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.4901376
time: 0.37 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2

### Relational analysis result of IS_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0069308, 0.7141405, 0.0789177, 0.4454548, -0.4523857, 0.6352228
1: -0.0711197, 0.9631418, 0.0311321, 0.5851125, -0.6562322, 0.9320097
2: -0.1773483, 0.7663486, -0.0367235, 0.5172547, -0.6946030, 0.8030721
3: -0.2441539, 0.9109904, -0.1111584, 0.5267107, -0.7708645, 1.0221487
4: -0.2841340, 1.0538890, -0.1149590, 0.6824017, -0.9665357, 1.1688480

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 39

Time for candidate selection: 4.40 seconds

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_B2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5341666, upper bound: 0.5618855
time: 0.35 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5426809, upper bound: 0.5606540
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0083361, 0.5719668, 0.0789177, 0.4454548, -0.4371188, 0.4930490
1: -0.0506135, 0.7433118, 0.0311321, 0.5851125, -0.6357260, 0.7121797
2: -0.1297052, 0.6455543, -0.0367235, 0.5172547, -0.6469599, 0.6822778
3: -0.2118729, 0.7126579, -0.1111584, 0.5267107, -0.7385836, 0.8238163
4: -0.2322460, 0.8428500, -0.1149590, 0.6824017, -0.9146477, 0.9578090

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 5

Time for candidate selection: 4.51 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_B2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5341666, upper bound: 0.5718158
time: 0.37 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5426809, upper bound: 0.5645301
time: 0.37 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 7.23 seconds
IS_A1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.23
Output dim: 0, lower bound: -0.5664861, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.23
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635
IS_A2_B1_B2_A1_B1, status: Status.VERIFIED, split count: 5, time: 7.23
Output dim: 0, lower bound: -0.5341666, upper bound: 0.5618855
IS_A2_B1_B2_A1_B2, status: Status.VERIFIED, split count: 5, time: 7.23
Output dim: 0, lower bound: -0.5426809, upper bound: 0.5606540
IS_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 7.23
Output dim: 0, lower bound: -0.5341666, upper bound: 0.5718158
IS_A2_B1_B2_A2_B2, status: Status.VERIFIED, split count: 5, time: 7.23
Output dim: 0, lower bound: -0.5426809, upper bound: 0.5645301

## BFS IS instance: IS_A1_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0896578, 0.4185404, -0.0208821, 0.5835277, -0.4938699, 0.4394225
1: 0.0485485, 0.5304989, -0.0742011, 0.8010436, -0.7524951, 0.6047000
2: -0.0087731, 0.5033746, -0.1870105, 0.6244919, -0.6332650, 0.6903851
3: -0.0878291, 0.4808285, -0.2587125, 0.7709957, -0.8588248, 0.7395410
4: -0.0913115, 0.6272116, -0.3087695, 0.8734686, -0.9647801, 0.9359810

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A1_B1

### Relational analysis result of IS_A1_A2_B1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
time: 0.35 seconds

## Relational analysis of IS_A1_A2_B1_B2_A1_B2

### Relational analysis result of IS_A1_A2_B1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
time: 0.38 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0862837, 0.4352814, -0.0208821, 0.5835277, -0.4972440, 0.4561635
1: 0.0415683, 0.5686141, -0.0742011, 0.8010436, -0.7594753, 0.6428152
2: -0.0245677, 0.5098621, -0.1870105, 0.6244919, -0.6490597, 0.6968727
3: -0.0975448, 0.5112243, -0.2587125, 0.7709957, -0.8685405, 0.7699367
4: -0.1047409, 0.6693002, -0.3087695, 0.8734686, -0.9782095, 0.9780697

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.5099635
time: 0.35 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0083361, 0.5719668, 0.0873206, 0.4219245, -0.4135884, 0.4846462
1: -0.0506135, 0.7433118, 0.0395762, 0.5401090, -0.5907225, 0.7037356
2: -0.1297052, 0.6455543, -0.0181572, 0.5037278, -0.6334330, 0.6637115
3: -0.2118729, 0.7126579, -0.0989394, 0.4913468, -0.7032197, 0.8115973
4: -0.2322460, 0.8428500, -0.0983677, 0.6463114, -0.8785574, 0.9412177

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5415571, upper bound: 0.5519340
time: 0.44 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5304913, upper bound: 0.5641851
time: 0.39 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 2.55 seconds
IS_A1_A2_B1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.55
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
IS_A1_A2_B1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.55
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.55
Output dim: 0, lower bound: -0.5494671, upper bound: 0.5099635
IS_A1_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635
IS_A2_B1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 6, time: 2.55
Output dim: 0, lower bound: -0.5415571, upper bound: 0.5519340
IS_A2_B1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 6, time: 2.55
Output dim: 0, lower bound: -0.5304913, upper bound: 0.5641851

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0862837, 0.4352814, -0.0208821, 0.5835277, -0.4972440, 0.4561635
1: 0.0415683, 0.5686141, -0.0742011, 0.8010436, -0.7594753, 0.6428152
2: -0.0245677, 0.5098621, -0.1870105, 0.6244919, -0.6490597, 0.6968727
3: -0.0975448, 0.5112243, -0.2587125, 0.7709957, -0.8685405, 0.7699367
4: -0.1047409, 0.6693002, -0.3087695, 0.8734686, -0.9782095, 0.9780697

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5460619, upper bound: 0.5060034
time: 0.38 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.4901376
time: 0.37 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635
time: 0.39 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 5.01 seconds
IS_A1_A2_B1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 0, lower bound: -0.5664861, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.01
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0903358, 0.4177946, -0.0208821, 0.5835277, -0.4931918, 0.4386767
1: 0.0493972, 0.5291730, -0.0742011, 0.8010436, -0.7516463, 0.6033741
2: -0.0079007, 0.5030103, -0.1870105, 0.6244919, -0.6323926, 0.6900208
3: -0.0871024, 0.4797198, -0.2587125, 0.7709957, -0.8580981, 0.7384322
4: -0.0905757, 0.6263897, -0.3087695, 0.8734686, -0.9640443, 0.9351592

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
time: 0.36 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
time: 0.38 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0862837, 0.4352814, -0.0208821, 0.5835277, -0.4972440, 0.4561635
1: 0.0415683, 0.5686141, -0.0742011, 0.8010436, -0.7594753, 0.6428152
2: -0.0245677, 0.5098621, -0.1870105, 0.6244919, -0.6490597, 0.6968727
3: -0.0975448, 0.5112243, -0.2587125, 0.7709957, -0.8685405, 0.7699367
4: -0.1047409, 0.6693002, -0.3087695, 0.8734686, -0.9782095, 0.9780697

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.5099635
time: 0.36 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635
time: 0.40 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 4.22 seconds
IS_A1_A2_B1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 4.22
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 4.22
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.22
Output dim: 0, lower bound: -0.5494671, upper bound: 0.5099635
IS_A1_A2_B1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0862837, 0.4352814, -0.0208821, 0.5835277, -0.4972440, 0.4561635
1: 0.0415683, 0.5686141, -0.0742011, 0.8010436, -0.7594753, 0.6428152
2: -0.0245677, 0.5098621, -0.1870105, 0.6244919, -0.6490597, 0.6968727
3: -0.0975448, 0.5112243, -0.2587125, 0.7709957, -0.8685405, 0.7699367
4: -0.1047409, 0.6693002, -0.3087695, 0.8734686, -0.9782095, 0.9780697

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5460619, upper bound: 0.5060034
time: 0.38 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.4901376
time: 0.41 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635
time: 0.39 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 5.06 seconds
IS_A1_A2_B1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 5.06
Output dim: 0, lower bound: -0.5664861, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 5.06
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0903358, 0.4177946, -0.0208821, 0.5835277, -0.4931918, 0.4386767
1: 0.0493972, 0.5291730, -0.0742011, 0.8010436, -0.7516463, 0.6033741
2: -0.0079007, 0.5030103, -0.1870105, 0.6244919, -0.6323926, 0.6900208
3: -0.0871024, 0.4797198, -0.2587125, 0.7709957, -0.8580981, 0.7384322
4: -0.0905757, 0.6263897, -0.3087695, 0.8734686, -0.9640443, 0.9351592

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
time: 0.37 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
time: 0.39 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0862837, 0.4352814, -0.0208821, 0.5835277, -0.4972440, 0.4561635
1: 0.0415683, 0.5686141, -0.0742011, 0.8010436, -0.7594753, 0.6428152
2: -0.0245677, 0.5098621, -0.1870105, 0.6244919, -0.6490597, 0.6968727
3: -0.0975448, 0.5112243, -0.2587125, 0.7709957, -0.8685405, 0.7699367
4: -0.1047409, 0.6693002, -0.3087695, 0.8734686, -0.9782095, 0.9780697

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.5099635
time: 0.36 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635
time: 0.39 seconds

## Summary of splitting at layer (split count: 9)
- Time for IS candidates: 4.22 seconds
IS_A1_A2_B1_B2_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 10, time: 4.22
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 10, time: 4.22
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 4.22
Output dim: 0, lower bound: -0.5494671, upper bound: 0.5099635
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.22
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0862837, 0.4352814, -0.0208821, 0.5835277, -0.4972440, 0.4561635
1: 0.0415683, 0.5686141, -0.0742011, 0.8010436, -0.7594753, 0.6428152
2: -0.0245677, 0.5098621, -0.1870105, 0.6244919, -0.6490597, 0.6968727
3: -0.0975448, 0.5112243, -0.2587125, 0.7709957, -0.8685405, 0.7699367
4: -0.1047409, 0.6693002, -0.3087695, 0.8734686, -0.9782095, 0.9780697

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5460619, upper bound: 0.5060034
time: 0.38 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.4901376
time: 0.41 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635
time: 0.38 seconds

## Summary of splitting at layer (split count: 10)
- Time for IS candidates: 5.12 seconds
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 11, time: 5.12
Output dim: 0, lower bound: -0.5664861, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 5.12
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0903358, 0.4177946, -0.0208821, 0.5835277, -0.4931918, 0.4386767
1: 0.0493972, 0.5291730, -0.0742011, 0.8010436, -0.7516463, 0.6033741
2: -0.0079007, 0.5030103, -0.1870105, 0.6244919, -0.6323926, 0.6900208
3: -0.0871024, 0.4797198, -0.2587125, 0.7709957, -0.8580981, 0.7384322
4: -0.0905757, 0.6263897, -0.3087695, 0.8734686, -0.9640443, 0.9351592

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
time: 0.43 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
time: 0.40 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0862837, 0.4352814, -0.0208821, 0.5835277, -0.4972440, 0.4561635
1: 0.0415683, 0.5686141, -0.0742011, 0.8010436, -0.7594753, 0.6428152
2: -0.0245677, 0.5098621, -0.1870105, 0.6244919, -0.6490597, 0.6968727
3: -0.0975448, 0.5112243, -0.2587125, 0.7709957, -0.8685405, 0.7699367
4: -0.1047409, 0.6693002, -0.3087695, 0.8734686, -0.9782095, 0.9780697

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.5099635
time: 0.40 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635
time: 0.40 seconds

## Summary of splitting at layer (split count: 11)
- Time for IS candidates: 4.33 seconds
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 12, time: 4.33
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 12, time: 4.33
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 12, time: 4.33
Output dim: 0, lower bound: -0.5494671, upper bound: 0.5099635
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 4.33
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0862837, 0.4352814, -0.0208821, 0.5835277, -0.4972440, 0.4561635
1: 0.0415683, 0.5686141, -0.0742011, 0.8010436, -0.7594753, 0.6428152
2: -0.0245677, 0.5098621, -0.1870105, 0.6244919, -0.6490597, 0.6968727
3: -0.0975448, 0.5112243, -0.2587125, 0.7709957, -0.8685405, 0.7699367
4: -0.1047409, 0.6693002, -0.3087695, 0.8734686, -0.9782095, 0.9780697

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5460619, upper bound: 0.5060034
time: 0.40 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.4901376
time: 0.42 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635
time: 0.39 seconds

## Summary of splitting at layer (split count: 12)
- Time for IS candidates: 5.30 seconds
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 13, time: 5.30
Output dim: 0, lower bound: -0.5664861, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 13, time: 5.30
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0903358, 0.4177946, -0.0208821, 0.5835277, -0.4931918, 0.4386767
1: 0.0493972, 0.5291730, -0.0742011, 0.8010436, -0.7516463, 0.6033741
2: -0.0079007, 0.5030103, -0.1870105, 0.6244919, -0.6323926, 0.6900208
3: -0.0871024, 0.4797198, -0.2587125, 0.7709957, -0.8580981, 0.7384322
4: -0.0905757, 0.6263897, -0.3087695, 0.8734686, -0.9640443, 0.9351592

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
time: 0.42 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
time: 0.42 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0862837, 0.4352814, -0.0208821, 0.5835277, -0.4972440, 0.4561635
1: 0.0415683, 0.5686141, -0.0742011, 0.8010436, -0.7594753, 0.6428152
2: -0.0245677, 0.5098621, -0.1870105, 0.6244919, -0.6490597, 0.6968727
3: -0.0975448, 0.5112243, -0.2587125, 0.7709957, -0.8685405, 0.7699367
4: -0.1047409, 0.6693002, -0.3087695, 0.8734686, -0.9782095, 0.9780697

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.5099635
time: 0.41 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635
time: 0.41 seconds

## Summary of splitting at layer (split count: 13)
- Time for IS candidates: 4.43 seconds
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 14, time: 4.43
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 14, time: 4.43
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 14, time: 4.43
Output dim: 0, lower bound: -0.5494671, upper bound: 0.5099635
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 14, time: 4.43
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0862837, 0.4352814, -0.0208821, 0.5835277, -0.4972440, 0.4561635
1: 0.0415683, 0.5686141, -0.0742011, 0.8010436, -0.7594753, 0.6428152
2: -0.0245677, 0.5098621, -0.1870105, 0.6244919, -0.6490597, 0.6968727
3: -0.0975448, 0.5112243, -0.2587125, 0.7709957, -0.8685405, 0.7699367
4: -0.1047409, 0.6693002, -0.3087695, 0.8734686, -0.9782095, 0.9780697

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5460619, upper bound: 0.5060034
time: 0.41 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.4901376
time: 0.44 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635
time: 0.41 seconds

## Summary of splitting at layer (split count: 14)
- Time for IS candidates: 5.33 seconds
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 15, time: 5.33
Output dim: 0, lower bound: -0.5664861, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 15, time: 5.33
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0903358, 0.4177946, -0.0208821, 0.5835277, -0.4931918, 0.4386767
1: 0.0493972, 0.5291730, -0.0742011, 0.8010436, -0.7516463, 0.6033741
2: -0.0079007, 0.5030103, -0.1870105, 0.6244919, -0.6323926, 0.6900208
3: -0.0871024, 0.4797198, -0.2587125, 0.7709957, -0.8580981, 0.7384322
4: -0.0905757, 0.6263897, -0.3087695, 0.8734686, -0.9640443, 0.9351592

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
time: 0.43 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
time: 0.43 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0862837, 0.4352814, -0.0208821, 0.5835277, -0.4972440, 0.4561635
1: 0.0415683, 0.5686141, -0.0742011, 0.8010436, -0.7594753, 0.6428152
2: -0.0245677, 0.5098621, -0.1870105, 0.6244919, -0.6490597, 0.6968727
3: -0.0975448, 0.5112243, -0.2587125, 0.7709957, -0.8685405, 0.7699367
4: -0.1047409, 0.6693002, -0.3087695, 0.8734686, -0.9782095, 0.9780697

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.5099635
time: 0.42 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635
time: 0.41 seconds

## Summary of splitting at layer (split count: 15)
- Time for IS candidates: 4.50 seconds
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 16, time: 4.50
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 16, time: 4.50
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 16, time: 4.50
Output dim: 0, lower bound: -0.5494671, upper bound: 0.5099635
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 16, time: 4.50
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0862837, 0.4352814, -0.0208821, 0.5835277, -0.4972440, 0.4561635
1: 0.0415683, 0.5686141, -0.0742011, 0.8010436, -0.7594753, 0.6428152
2: -0.0245677, 0.5098621, -0.1870105, 0.6244919, -0.6490597, 0.6968727
3: -0.0975448, 0.5112243, -0.2587125, 0.7709957, -0.8685405, 0.7699367
4: -0.1047409, 0.6693002, -0.3087695, 0.8734686, -0.9782095, 0.9780697

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5460619, upper bound: 0.5060034
time: 0.42 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.4901376
time: 0.47 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635
time: 0.42 seconds

## Summary of splitting at layer (split count: 16)
- Time for IS candidates: 5.51 seconds
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 17, time: 5.51
Output dim: 0, lower bound: -0.5664861, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 17, time: 5.51
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0903358, 0.4177946, -0.0208821, 0.5835277, -0.4931918, 0.4386767
1: 0.0493972, 0.5291730, -0.0742011, 0.8010436, -0.7516463, 0.6033741
2: -0.0079007, 0.5030103, -0.1870105, 0.6244919, -0.6323926, 0.6900208
3: -0.0871024, 0.4797198, -0.2587125, 0.7709957, -0.8580981, 0.7384322
4: -0.0905757, 0.6263897, -0.3087695, 0.8734686, -0.9640443, 0.9351592

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
time: 0.45 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
time: 0.42 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0862837, 0.4352814, -0.0208821, 0.5835277, -0.4972440, 0.4561635
1: 0.0415683, 0.5686141, -0.0742011, 0.8010436, -0.7594753, 0.6428152
2: -0.0245677, 0.5098621, -0.1870105, 0.6244919, -0.6490597, 0.6968727
3: -0.0975448, 0.5112243, -0.2587125, 0.7709957, -0.8685405, 0.7699367
4: -0.1047409, 0.6693002, -0.3087695, 0.8734686, -0.9782095, 0.9780697

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.5099635
time: 0.44 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635
time: 0.42 seconds

## Summary of splitting at layer (split count: 17)
- Time for IS candidates: 4.54 seconds
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 18, time: 4.54
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 18, time: 4.54
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 18, time: 4.54
Output dim: 0, lower bound: -0.5494671, upper bound: 0.5099635
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 18, time: 4.54
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0862837, 0.4352814, -0.0208821, 0.5835277, -0.4972440, 0.4561635
1: 0.0415683, 0.5686141, -0.0742011, 0.8010436, -0.7594753, 0.6428152
2: -0.0245677, 0.5098621, -0.1870105, 0.6244919, -0.6490597, 0.6968727
3: -0.0975448, 0.5112243, -0.2587125, 0.7709957, -0.8685405, 0.7699367
4: -0.1047409, 0.6693002, -0.3087695, 0.8734686, -0.9782095, 0.9780697

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5460619, upper bound: 0.5060034
time: 0.43 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.4901376
time: 0.46 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635
time: 0.43 seconds

## Summary of splitting at layer (split count: 18)
- Time for IS candidates: 5.56 seconds
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 19, time: 5.56
Output dim: 0, lower bound: -0.5664861, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 19, time: 5.56
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0903358, 0.4177946, -0.0208821, 0.5835277, -0.4931918, 0.4386767
1: 0.0493972, 0.5291730, -0.0742011, 0.8010436, -0.7516463, 0.6033741
2: -0.0079007, 0.5030103, -0.1870105, 0.6244919, -0.6323926, 0.6900208
3: -0.0871024, 0.4797198, -0.2587125, 0.7709957, -0.8580981, 0.7384322
4: -0.0905757, 0.6263897, -0.3087695, 0.8734686, -0.9640443, 0.9351592

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
time: 0.45 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
time: 0.44 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0862837, 0.4352814, -0.0208821, 0.5835277, -0.4972440, 0.4561635
1: 0.0415683, 0.5686141, -0.0742011, 0.8010436, -0.7594753, 0.6428152
2: -0.0245677, 0.5098621, -0.1870105, 0.6244919, -0.6490597, 0.6968727
3: -0.0975448, 0.5112243, -0.2587125, 0.7709957, -0.8685405, 0.7699367
4: -0.1047409, 0.6693002, -0.3087695, 0.8734686, -0.9782095, 0.9780697

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.5099635
time: 0.45 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635
time: 0.44 seconds

## Summary of splitting at layer (split count: 19)
- Time for IS candidates: 4.62 seconds
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 20, time: 4.62
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 20, time: 4.62
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 20, time: 4.62
Output dim: 0, lower bound: -0.5494671, upper bound: 0.5099635
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 20, time: 4.62
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0862837, 0.4352814, -0.0208821, 0.5835277, -0.4972440, 0.4561635
1: 0.0415683, 0.5686141, -0.0742011, 0.8010436, -0.7594753, 0.6428152
2: -0.0245677, 0.5098621, -0.1870105, 0.6244919, -0.6490597, 0.6968727
3: -0.0975448, 0.5112243, -0.2587125, 0.7709957, -0.8685405, 0.7699367
4: -0.1047409, 0.6693002, -0.3087695, 0.8734686, -0.9782095, 0.9780697

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5460619, upper bound: 0.5060034
time: 0.44 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.4901376
time: 0.47 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635
time: 0.44 seconds

## Summary of splitting at layer (split count: 20)
- Time for IS candidates: 5.59 seconds
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 21, time: 5.59
Output dim: 0, lower bound: -0.5664861, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 21, time: 5.59
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0903358, 0.4177946, -0.0208821, 0.5835277, -0.4931918, 0.4386767
1: 0.0493972, 0.5291730, -0.0742011, 0.8010436, -0.7516463, 0.6033741
2: -0.0079007, 0.5030103, -0.1870105, 0.6244919, -0.6323926, 0.6900208
3: -0.0871024, 0.4797198, -0.2587125, 0.7709957, -0.8580981, 0.7384322
4: -0.0905757, 0.6263897, -0.3087695, 0.8734686, -0.9640443, 0.9351592

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
time: 0.43 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
time: 0.49 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0862837, 0.4352814, -0.0208821, 0.5835277, -0.4972440, 0.4561635
1: 0.0415683, 0.5686141, -0.0742011, 0.8010436, -0.7594753, 0.6428152
2: -0.0245677, 0.5098621, -0.1870105, 0.6244919, -0.6490597, 0.6968727
3: -0.0975448, 0.5112243, -0.2587125, 0.7709957, -0.8685405, 0.7699367
4: -0.1047409, 0.6693002, -0.3087695, 0.8734686, -0.9782095, 0.9780697

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.5099635
time: 0.46 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635
time: 0.47 seconds

## Summary of splitting at layer (split count: 21)
- Time for IS candidates: 4.74 seconds
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 22, time: 4.74
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 22, time: 4.74
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 22, time: 4.74
Output dim: 0, lower bound: -0.5494671, upper bound: 0.5099635
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 22, time: 4.74
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0862837, 0.4352814, -0.0208821, 0.5835277, -0.4972440, 0.4561635
1: 0.0415683, 0.5686141, -0.0742011, 0.8010436, -0.7594753, 0.6428152
2: -0.0245677, 0.5098621, -0.1870105, 0.6244919, -0.6490597, 0.6968727
3: -0.0975448, 0.5112243, -0.2587125, 0.7709957, -0.8685405, 0.7699367
4: -0.1047409, 0.6693002, -0.3087695, 0.8734686, -0.9782095, 0.9780697

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5460619, upper bound: 0.5060034
time: 0.45 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.4901376
time: 0.47 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635
time: 0.46 seconds

## Summary of splitting at layer (split count: 22)
- Time for IS candidates: 5.71 seconds
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 23, time: 5.71
Output dim: 0, lower bound: -0.5664861, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 23, time: 5.71
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0903358, 0.4177946, -0.0208821, 0.5835277, -0.4931918, 0.4386767
1: 0.0493972, 0.5291730, -0.0742011, 0.8010436, -0.7516463, 0.6033741
2: -0.0079007, 0.5030103, -0.1870105, 0.6244919, -0.6323926, 0.6900208
3: -0.0871024, 0.4797198, -0.2587125, 0.7709957, -0.8580981, 0.7384322
4: -0.0905757, 0.6263897, -0.3087695, 0.8734686, -0.9640443, 0.9351592

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
time: 0.46 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
time: 0.50 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0862837, 0.4352814, -0.0208821, 0.5835277, -0.4972440, 0.4561635
1: 0.0415683, 0.5686141, -0.0742011, 0.8010436, -0.7594753, 0.6428152
2: -0.0245677, 0.5098621, -0.1870105, 0.6244919, -0.6490597, 0.6968727
3: -0.0975448, 0.5112243, -0.2587125, 0.7709957, -0.8685405, 0.7699367
4: -0.1047409, 0.6693002, -0.3087695, 0.8734686, -0.9782095, 0.9780697

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.5099635
time: 0.47 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635
time: 0.48 seconds

## Summary of splitting at layer (split count: 23)
- Time for IS candidates: 4.80 seconds
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 24, time: 4.80
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 24, time: 4.80
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 24, time: 4.80
Output dim: 0, lower bound: -0.5494671, upper bound: 0.5099635
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 24, time: 4.80
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0862837, 0.4352814, -0.0208821, 0.5835277, -0.4972440, 0.4561635
1: 0.0415683, 0.5686141, -0.0742011, 0.8010436, -0.7594753, 0.6428152
2: -0.0245677, 0.5098621, -0.1870105, 0.6244919, -0.6490597, 0.6968727
3: -0.0975448, 0.5112243, -0.2587125, 0.7709957, -0.8685405, 0.7699367
4: -0.1047409, 0.6693002, -0.3087695, 0.8734686, -0.9782095, 0.9780697

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5460619, upper bound: 0.5060034
time: 0.46 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.4901376
time: 0.49 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635
time: 0.46 seconds

## Summary of splitting at layer (split count: 24)
- Time for IS candidates: 5.87 seconds
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 25, time: 5.87
Output dim: 0, lower bound: -0.5664861, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 25, time: 5.87
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0903358, 0.4177946, -0.0208821, 0.5835277, -0.4931918, 0.4386767
1: 0.0493972, 0.5291730, -0.0742011, 0.8010436, -0.7516463, 0.6033741
2: -0.0079007, 0.5030103, -0.1870105, 0.6244919, -0.6323926, 0.6900208
3: -0.0871024, 0.4797198, -0.2587125, 0.7709957, -0.8580981, 0.7384322
4: -0.0905757, 0.6263897, -0.3087695, 0.8734686, -0.9640443, 0.9351592

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
time: 0.47 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
time: 0.52 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0862837, 0.4352814, -0.0208821, 0.5835277, -0.4972440, 0.4561635
1: 0.0415683, 0.5686141, -0.0742011, 0.8010436, -0.7594753, 0.6428152
2: -0.0245677, 0.5098621, -0.1870105, 0.6244919, -0.6490597, 0.6968727
3: -0.0975448, 0.5112243, -0.2587125, 0.7709957, -0.8685405, 0.7699367
4: -0.1047409, 0.6693002, -0.3087695, 0.8734686, -0.9782095, 0.9780697

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.5099635
time: 0.47 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635
time: 0.47 seconds

## Summary of splitting at layer (split count: 25)
- Time for IS candidates: 4.88 seconds
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 26, time: 4.88
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 26, time: 4.88
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 26, time: 4.88
Output dim: 0, lower bound: -0.5494671, upper bound: 0.5099635
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 26, time: 4.88
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0862837, 0.4352814, -0.0208821, 0.5835277, -0.4972440, 0.4561635
1: 0.0415683, 0.5686141, -0.0742011, 0.8010436, -0.7594753, 0.6428152
2: -0.0245677, 0.5098621, -0.1870105, 0.6244919, -0.6490597, 0.6968727
3: -0.0975448, 0.5112243, -0.2587125, 0.7709957, -0.8685405, 0.7699367
4: -0.1047409, 0.6693002, -0.3087695, 0.8734686, -0.9782095, 0.9780697

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5460619, upper bound: 0.5060034
time: 0.47 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.4901376
time: 0.46 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635
time: 0.47 seconds

## Summary of splitting at layer (split count: 26)
- Time for IS candidates: 5.84 seconds
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 27, time: 5.84
Output dim: 0, lower bound: -0.5664861, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 27, time: 5.84
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0903358, 0.4177946, -0.0208821, 0.5835277, -0.4931918, 0.4386767
1: 0.0493972, 0.5291730, -0.0742011, 0.8010436, -0.7516463, 0.6033741
2: -0.0079007, 0.5030103, -0.1870105, 0.6244919, -0.6323926, 0.6900208
3: -0.0871024, 0.4797198, -0.2587125, 0.7709957, -0.8580981, 0.7384322
4: -0.0905757, 0.6263897, -0.3087695, 0.8734686, -0.9640443, 0.9351592

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
time: 0.46 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
time: 0.53 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0862837, 0.4352814, -0.0208821, 0.5835277, -0.4972440, 0.4561635
1: 0.0415683, 0.5686141, -0.0742011, 0.8010436, -0.7594753, 0.6428152
2: -0.0245677, 0.5098621, -0.1870105, 0.6244919, -0.6490597, 0.6968727
3: -0.0975448, 0.5112243, -0.2587125, 0.7709957, -0.8685405, 0.7699367
4: -0.1047409, 0.6693002, -0.3087695, 0.8734686, -0.9782095, 0.9780697

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.5099635
time: 0.48 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635
time: 0.49 seconds

## Summary of splitting at layer (split count: 27)
- Time for IS candidates: 4.96 seconds
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 28, time: 4.96
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 28, time: 4.96
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 28, time: 4.96
Output dim: 0, lower bound: -0.5494671, upper bound: 0.5099635
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 28, time: 4.96
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0862837, 0.4352814, -0.0208821, 0.5835277, -0.4972440, 0.4561635
1: 0.0415683, 0.5686141, -0.0742011, 0.8010436, -0.7594753, 0.6428152
2: -0.0245677, 0.5098621, -0.1870105, 0.6244919, -0.6490597, 0.6968727
3: -0.0975448, 0.5112243, -0.2587125, 0.7709957, -0.8685405, 0.7699367
4: -0.1047409, 0.6693002, -0.3087695, 0.8734686, -0.9782095, 0.9780697

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5460619, upper bound: 0.5060034
time: 0.47 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.4901376
time: 0.47 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635
time: 0.48 seconds

## Summary of splitting at layer (split count: 28)
- Time for IS candidates: 5.96 seconds
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 29, time: 5.96
Output dim: 0, lower bound: -0.5664861, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 29, time: 5.96
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0903358, 0.4177946, -0.0208821, 0.5835277, -0.4931918, 0.4386767
1: 0.0493972, 0.5291730, -0.0742011, 0.8010436, -0.7516463, 0.6033741
2: -0.0079007, 0.5030103, -0.1870105, 0.6244919, -0.6323926, 0.6900208
3: -0.0871024, 0.4797198, -0.2587125, 0.7709957, -0.8580981, 0.7384322
4: -0.0905757, 0.6263897, -0.3087695, 0.8734686, -0.9640443, 0.9351592

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
time: 0.47 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
time: 0.54 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0862837, 0.4352814, -0.0208821, 0.5835277, -0.4972440, 0.4561635
1: 0.0415683, 0.5686141, -0.0742011, 0.8010436, -0.7594753, 0.6428152
2: -0.0245677, 0.5098621, -0.1870105, 0.6244919, -0.6490597, 0.6968727
3: -0.0975448, 0.5112243, -0.2587125, 0.7709957, -0.8685405, 0.7699367
4: -0.1047409, 0.6693002, -0.3087695, 0.8734686, -0.9782095, 0.9780697

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.5099635
time: 0.49 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635
time: 0.49 seconds

## Summary of splitting at layer (split count: 29)
- Time for IS candidates: 5.01 seconds
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 30, time: 5.01
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 30, time: 5.01
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 30, time: 5.01
Output dim: 0, lower bound: -0.5494671, upper bound: 0.5099635
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 30, time: 5.01
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0862837, 0.4352814, -0.0208821, 0.5835277, -0.4972440, 0.4561635
1: 0.0415683, 0.5686141, -0.0742011, 0.8010436, -0.7594753, 0.6428152
2: -0.0245677, 0.5098621, -0.1870105, 0.6244919, -0.6490597, 0.6968727
3: -0.0975448, 0.5112243, -0.2587125, 0.7709957, -0.8685405, 0.7699367
4: -0.1047409, 0.6693002, -0.3087695, 0.8734686, -0.9782095, 0.9780697

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5460619, upper bound: 0.5060034
time: 0.47 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.4901376
time: 0.48 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635
time: 0.50 seconds

## Summary of splitting at layer (split count: 30)
- Time for IS candidates: 6.03 seconds
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 31, time: 6.03
Output dim: 0, lower bound: -0.5664861, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 31, time: 6.03
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0903358, 0.4177946, -0.0208821, 0.5835277, -0.4931918, 0.4386767
1: 0.0493972, 0.5291730, -0.0742011, 0.8010436, -0.7516463, 0.6033741
2: -0.0079007, 0.5030103, -0.1870105, 0.6244919, -0.6323926, 0.6900208
3: -0.0871024, 0.4797198, -0.2587125, 0.7709957, -0.8580981, 0.7384322
4: -0.0905757, 0.6263897, -0.3087695, 0.8734686, -0.9640443, 0.9351592

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
time: 0.48 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
time: 0.50 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0862837, 0.4352814, -0.0208821, 0.5835277, -0.4972440, 0.4561635
1: 0.0415683, 0.5686141, -0.0742011, 0.8010436, -0.7594753, 0.6428152
2: -0.0245677, 0.5098621, -0.1870105, 0.6244919, -0.6490597, 0.6968727
3: -0.0975448, 0.5112243, -0.2587125, 0.7709957, -0.8685405, 0.7699367
4: -0.1047409, 0.6693002, -0.3087695, 0.8734686, -0.9782095, 0.9780697

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.5099635
time: 0.50 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635
time: 0.51 seconds

## Summary of splitting at layer (split count: 31)
- Time for IS candidates: 5.01 seconds
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 32, time: 5.01
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 32, time: 5.01
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 32, time: 5.01
Output dim: 0, lower bound: -0.5494671, upper bound: 0.5099635
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 32, time: 5.01
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0862837, 0.4352814, -0.0208821, 0.5835277, -0.4972440, 0.4561635
1: 0.0415683, 0.5686141, -0.0742011, 0.8010436, -0.7594753, 0.6428152
2: -0.0245677, 0.5098621, -0.1870105, 0.6244919, -0.6490597, 0.6968727
3: -0.0975448, 0.5112243, -0.2587125, 0.7709957, -0.8685405, 0.7699367
4: -0.1047409, 0.6693002, -0.3087695, 0.8734686, -0.9782095, 0.9780697

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5460619, upper bound: 0.5060034
time: 0.49 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.4901376
time: 0.49 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635
time: 0.51 seconds

## Summary of splitting at layer (split count: 32)
- Time for IS candidates: 6.07 seconds
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 33, time: 6.07
Output dim: 0, lower bound: -0.5664861, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 33, time: 6.07
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0903358, 0.4177946, -0.0208821, 0.5835277, -0.4931918, 0.4386767
1: 0.0493972, 0.5291730, -0.0742011, 0.8010436, -0.7516463, 0.6033741
2: -0.0079007, 0.5030103, -0.1870105, 0.6244919, -0.6323926, 0.6900208
3: -0.0871024, 0.4797198, -0.2587125, 0.7709957, -0.8580981, 0.7384322
4: -0.0905757, 0.6263897, -0.3087695, 0.8734686, -0.9640443, 0.9351592

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
time: 0.49 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
time: 0.50 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0862837, 0.4352814, -0.0208821, 0.5835277, -0.4972440, 0.4561635
1: 0.0415683, 0.5686141, -0.0742011, 0.8010436, -0.7594753, 0.6428152
2: -0.0245677, 0.5098621, -0.1870105, 0.6244919, -0.6490597, 0.6968727
3: -0.0975448, 0.5112243, -0.2587125, 0.7709957, -0.8685405, 0.7699367
4: -0.1047409, 0.6693002, -0.3087695, 0.8734686, -0.9782095, 0.9780697

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.5099635
time: 0.50 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635
time: 0.52 seconds

## Summary of splitting at layer (split count: 33)
- Time for IS candidates: 5.08 seconds
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 34, time: 5.08
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 34, time: 5.08
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 34, time: 5.08
Output dim: 0, lower bound: -0.5494671, upper bound: 0.5099635
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 34, time: 5.08
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0862837, 0.4352814, -0.0208821, 0.5835277, -0.4972440, 0.4561635
1: 0.0415683, 0.5686141, -0.0742011, 0.8010436, -0.7594753, 0.6428152
2: -0.0245677, 0.5098621, -0.1870105, 0.6244919, -0.6490597, 0.6968727
3: -0.0975448, 0.5112243, -0.2587125, 0.7709957, -0.8685405, 0.7699367
4: -0.1047409, 0.6693002, -0.3087695, 0.8734686, -0.9782095, 0.9780697

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5460619, upper bound: 0.5060034
time: 0.49 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.4901376
time: 0.50 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635
time: 0.52 seconds

## Summary of splitting at layer (split count: 34)
- Time for IS candidates: 6.20 seconds
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 35, time: 6.20
Output dim: 0, lower bound: -0.5664861, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 35, time: 6.20
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0903358, 0.4177946, -0.0208821, 0.5835277, -0.4931918, 0.4386767
1: 0.0493972, 0.5291730, -0.0742011, 0.8010436, -0.7516463, 0.6033741
2: -0.0079007, 0.5030103, -0.1870105, 0.6244919, -0.6323926, 0.6900208
3: -0.0871024, 0.4797198, -0.2587125, 0.7709957, -0.8580981, 0.7384322
4: -0.0905757, 0.6263897, -0.3087695, 0.8734686, -0.9640443, 0.9351592

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
time: 0.53 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
time: 0.51 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0862837, 0.4352814, -0.0208821, 0.5835277, -0.4972440, 0.4561635
1: 0.0415683, 0.5686141, -0.0742011, 0.8010436, -0.7594753, 0.6428152
2: -0.0245677, 0.5098621, -0.1870105, 0.6244919, -0.6490597, 0.6968727
3: -0.0975448, 0.5112243, -0.2587125, 0.7709957, -0.8685405, 0.7699367
4: -0.1047409, 0.6693002, -0.3087695, 0.8734686, -0.9782095, 0.9780697

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.5099635
time: 0.52 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635
time: 0.54 seconds

## Summary of splitting at layer (split count: 35)
- Time for IS candidates: 5.14 seconds
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 36, time: 5.14
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 36, time: 5.14
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 36, time: 5.14
Output dim: 0, lower bound: -0.5494671, upper bound: 0.5099635
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 36, time: 5.14
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0862837, 0.4352814, -0.0208821, 0.5835277, -0.4972440, 0.4561635
1: 0.0415683, 0.5686141, -0.0742011, 0.8010436, -0.7594753, 0.6428152
2: -0.0245677, 0.5098621, -0.1870105, 0.6244919, -0.6490597, 0.6968727
3: -0.0975448, 0.5112243, -0.2587125, 0.7709957, -0.8685405, 0.7699367
4: -0.1047409, 0.6693002, -0.3087695, 0.8734686, -0.9782095, 0.9780697

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5460619, upper bound: 0.5060034
time: 0.50 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.4901376
time: 0.51 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635
time: 0.52 seconds

## Summary of splitting at layer (split count: 36)
- Time for IS candidates: 6.29 seconds
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 37, time: 6.29
Output dim: 0, lower bound: -0.5664861, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 37, time: 6.29
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0903358, 0.4177946, -0.0208821, 0.5835277, -0.4931918, 0.4386767
1: 0.0493972, 0.5291730, -0.0742011, 0.8010436, -0.7516463, 0.6033741
2: -0.0079007, 0.5030103, -0.1870105, 0.6244919, -0.6323926, 0.6900208
3: -0.0871024, 0.4797198, -0.2587125, 0.7709957, -0.8580981, 0.7384322
4: -0.0905757, 0.6263897, -0.3087695, 0.8734686, -0.9640443, 0.9351592

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
time: 0.56 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
time: 0.52 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0862837, 0.4352814, -0.0208821, 0.5835277, -0.4972440, 0.4561635
1: 0.0415683, 0.5686141, -0.0742011, 0.8010436, -0.7594753, 0.6428152
2: -0.0245677, 0.5098621, -0.1870105, 0.6244919, -0.6490597, 0.6968727
3: -0.0975448, 0.5112243, -0.2587125, 0.7709957, -0.8685405, 0.7699367
4: -0.1047409, 0.6693002, -0.3087695, 0.8734686, -0.9782095, 0.9780697

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.5099635
time: 0.53 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635
time: 0.56 seconds

## Summary of splitting at layer (split count: 37)
- Time for IS candidates: 5.25 seconds
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 38, time: 5.25
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 38, time: 5.25
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 38, time: 5.25
Output dim: 0, lower bound: -0.5494671, upper bound: 0.5099635
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 38, time: 5.25
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0862837, 0.4352814, -0.0208821, 0.5835277, -0.4972440, 0.4561635
1: 0.0415683, 0.5686141, -0.0742011, 0.8010436, -0.7594753, 0.6428152
2: -0.0245677, 0.5098621, -0.1870105, 0.6244919, -0.6490597, 0.6968727
3: -0.0975448, 0.5112243, -0.2587125, 0.7709957, -0.8685405, 0.7699367
4: -0.1047409, 0.6693002, -0.3087695, 0.8734686, -0.9782095, 0.9780697

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5460619, upper bound: 0.5060034
time: 0.51 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.4901376
time: 0.52 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635
time: 0.53 seconds

## Summary of splitting at layer (split count: 38)
- Time for IS candidates: 6.42 seconds
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 39, time: 6.42
Output dim: 0, lower bound: -0.5664861, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 39, time: 6.42
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0903358, 0.4177946, -0.0208821, 0.5835277, -0.4931918, 0.4386767
1: 0.0493972, 0.5291730, -0.0742011, 0.8010436, -0.7516463, 0.6033741
2: -0.0079007, 0.5030103, -0.1870105, 0.6244919, -0.6323926, 0.6900208
3: -0.0871024, 0.4797198, -0.2587125, 0.7709957, -0.8580981, 0.7384322
4: -0.0905757, 0.6263897, -0.3087695, 0.8734686, -0.9640443, 0.9351592

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
time: 0.57 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
time: 0.54 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0862837, 0.4352814, -0.0208821, 0.5835277, -0.4972440, 0.4561635
1: 0.0415683, 0.5686141, -0.0742011, 0.8010436, -0.7594753, 0.6428152
2: -0.0245677, 0.5098621, -0.1870105, 0.6244919, -0.6490597, 0.6968727
3: -0.0975448, 0.5112243, -0.2587125, 0.7709957, -0.8685405, 0.7699367
4: -0.1047409, 0.6693002, -0.3087695, 0.8734686, -0.9782095, 0.9780697

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.5099635
time: 0.54 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635
time: 0.54 seconds

## Summary of splitting at layer (split count: 39)
- Time for IS candidates: 5.33 seconds
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 40, time: 5.33
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 40, time: 5.33
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 40, time: 5.33
Output dim: 0, lower bound: -0.5494671, upper bound: 0.5099635
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 40, time: 5.33
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0862837, 0.4352814, -0.0208821, 0.5835277, -0.4972440, 0.4561635
1: 0.0415683, 0.5686141, -0.0742011, 0.8010436, -0.7594753, 0.6428152
2: -0.0245677, 0.5098621, -0.1870105, 0.6244919, -0.6490597, 0.6968727
3: -0.0975448, 0.5112243, -0.2587125, 0.7709957, -0.8685405, 0.7699367
4: -0.1047409, 0.6693002, -0.3087695, 0.8734686, -0.9782095, 0.9780697

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5460619, upper bound: 0.5060034
time: 0.53 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.4901376
time: 0.54 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635
time: 0.55 seconds

## Summary of splitting at layer (split count: 40)
- Time for IS candidates: 6.49 seconds
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 41, time: 6.49
Output dim: 0, lower bound: -0.5664861, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 41, time: 6.49
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0903358, 0.4177946, -0.0208821, 0.5835277, -0.4931918, 0.4386767
1: 0.0493972, 0.5291730, -0.0742011, 0.8010436, -0.7516463, 0.6033741
2: -0.0079007, 0.5030103, -0.1870105, 0.6244919, -0.6323926, 0.6900208
3: -0.0871024, 0.4797198, -0.2587125, 0.7709957, -0.8580981, 0.7384322
4: -0.0905757, 0.6263897, -0.3087695, 0.8734686, -0.9640443, 0.9351592

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
time: 0.56 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
time: 0.57 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0862837, 0.4352814, -0.0208821, 0.5835277, -0.4972440, 0.4561635
1: 0.0415683, 0.5686141, -0.0742011, 0.8010436, -0.7594753, 0.6428152
2: -0.0245677, 0.5098621, -0.1870105, 0.6244919, -0.6490597, 0.6968727
3: -0.0975448, 0.5112243, -0.2587125, 0.7709957, -0.8685405, 0.7699367
4: -0.1047409, 0.6693002, -0.3087695, 0.8734686, -0.9782095, 0.9780697

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.5099635
time: 0.55 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635
time: 0.56 seconds

## Summary of splitting at layer (split count: 41)
- Time for IS candidates: 5.35 seconds
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 42, time: 5.35
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 42, time: 5.35
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 42, time: 5.35
Output dim: 0, lower bound: -0.5494671, upper bound: 0.5099635
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 42, time: 5.35
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0862837, 0.4352814, -0.0208821, 0.5835277, -0.4972440, 0.4561635
1: 0.0415683, 0.5686141, -0.0742011, 0.8010436, -0.7594753, 0.6428152
2: -0.0245677, 0.5098621, -0.1870105, 0.6244919, -0.6490597, 0.6968727
3: -0.0975448, 0.5112243, -0.2587125, 0.7709957, -0.8685405, 0.7699367
4: -0.1047409, 0.6693002, -0.3087695, 0.8734686, -0.9782095, 0.9780697

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5460619, upper bound: 0.5060034
time: 0.54 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.4901376
time: 0.54 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635
time: 0.56 seconds

## Summary of splitting at layer (split count: 42)
- Time for IS candidates: 6.58 seconds
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 43, time: 6.58
Output dim: 0, lower bound: -0.5664861, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 43, time: 6.58
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0903358, 0.4177946, -0.0208821, 0.5835277, -0.4931918, 0.4386767
1: 0.0493972, 0.5291730, -0.0742011, 0.8010436, -0.7516463, 0.6033741
2: -0.0079007, 0.5030103, -0.1870105, 0.6244919, -0.6323926, 0.6900208
3: -0.0871024, 0.4797198, -0.2587125, 0.7709957, -0.8580981, 0.7384322
4: -0.0905757, 0.6263897, -0.3087695, 0.8734686, -0.9640443, 0.9351592

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
time: 0.61 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
time: 0.56 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0862837, 0.4352814, -0.0208821, 0.5835277, -0.4972440, 0.4561635
1: 0.0415683, 0.5686141, -0.0742011, 0.8010436, -0.7594753, 0.6428152
2: -0.0245677, 0.5098621, -0.1870105, 0.6244919, -0.6490597, 0.6968727
3: -0.0975448, 0.5112243, -0.2587125, 0.7709957, -0.8685405, 0.7699367
4: -0.1047409, 0.6693002, -0.3087695, 0.8734686, -0.9782095, 0.9780697

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.5099635
time: 0.56 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635
time: 0.57 seconds

## Summary of splitting at layer (split count: 43)
- Time for IS candidates: 5.48 seconds
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 44, time: 5.48
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 44, time: 5.48
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 44, time: 5.48
Output dim: 0, lower bound: -0.5494671, upper bound: 0.5099635
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 44, time: 5.48
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0862837, 0.4352814, -0.0208821, 0.5835277, -0.4972440, 0.4561635
1: 0.0415683, 0.5686141, -0.0742011, 0.8010436, -0.7594753, 0.6428152
2: -0.0245677, 0.5098621, -0.1870105, 0.6244919, -0.6490597, 0.6968727
3: -0.0975448, 0.5112243, -0.2587125, 0.7709957, -0.8685405, 0.7699367
4: -0.1047409, 0.6693002, -0.3087695, 0.8734686, -0.9782095, 0.9780697

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5460619, upper bound: 0.5060034
time: 0.56 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.4901376
time: 0.55 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635
time: 0.58 seconds

## Summary of splitting at layer (split count: 44)
- Time for IS candidates: 6.71 seconds
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 45, time: 6.71
Output dim: 0, lower bound: -0.5664861, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 45, time: 6.71
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0903358, 0.4177946, -0.0208821, 0.5835277, -0.4931918, 0.4386767
1: 0.0493972, 0.5291730, -0.0742011, 0.8010436, -0.7516463, 0.6033741
2: -0.0079007, 0.5030103, -0.1870105, 0.6244919, -0.6323926, 0.6900208
3: -0.0871024, 0.4797198, -0.2587125, 0.7709957, -0.8580981, 0.7384322
4: -0.0905757, 0.6263897, -0.3087695, 0.8734686, -0.9640443, 0.9351592

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
time: 0.58 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
time: 0.56 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0862837, 0.4352814, -0.0208821, 0.5835277, -0.4972440, 0.4561635
1: 0.0415683, 0.5686141, -0.0742011, 0.8010436, -0.7594753, 0.6428152
2: -0.0245677, 0.5098621, -0.1870105, 0.6244919, -0.6490597, 0.6968727
3: -0.0975448, 0.5112243, -0.2587125, 0.7709957, -0.8685405, 0.7699367
4: -0.1047409, 0.6693002, -0.3087695, 0.8734686, -0.9782095, 0.9780697

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.5099635
time: 0.56 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635
time: 0.62 seconds

## Summary of splitting at layer (split count: 45)
- Time for IS candidates: 5.63 seconds
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 46, time: 5.63
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 46, time: 5.63
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 46, time: 5.63
Output dim: 0, lower bound: -0.5494671, upper bound: 0.5099635
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 46, time: 5.63
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0862837, 0.4352814, -0.0208821, 0.5835277, -0.4972440, 0.4561635
1: 0.0415683, 0.5686141, -0.0742011, 0.8010436, -0.7594753, 0.6428152
2: -0.0245677, 0.5098621, -0.1870105, 0.6244919, -0.6490597, 0.6968727
3: -0.0975448, 0.5112243, -0.2587125, 0.7709957, -0.8685405, 0.7699367
4: -0.1047409, 0.6693002, -0.3087695, 0.8734686, -0.9782095, 0.9780697

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5460619, upper bound: 0.5060034
time: 0.56 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.4901376
time: 0.56 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635
time: 0.58 seconds

## Summary of splitting at layer (split count: 46)
- Time for IS candidates: 6.89 seconds
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 47, time: 6.89
Output dim: 0, lower bound: -0.5664861, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 47, time: 6.89
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0903358, 0.4177946, -0.0208821, 0.5835277, -0.4931918, 0.4386767
1: 0.0493972, 0.5291730, -0.0742011, 0.8010436, -0.7516463, 0.6033741
2: -0.0079007, 0.5030103, -0.1870105, 0.6244919, -0.6323926, 0.6900208
3: -0.0871024, 0.4797198, -0.2587125, 0.7709957, -0.8580981, 0.7384322
4: -0.0905757, 0.6263897, -0.3087695, 0.8734686, -0.9640443, 0.9351592

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
time: 0.60 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
time: 0.59 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0862837, 0.4352814, -0.0208821, 0.5835277, -0.4972440, 0.4561635
1: 0.0415683, 0.5686141, -0.0742011, 0.8010436, -0.7594753, 0.6428152
2: -0.0245677, 0.5098621, -0.1870105, 0.6244919, -0.6490597, 0.6968727
3: -0.0975448, 0.5112243, -0.2587125, 0.7709957, -0.8685405, 0.7699367
4: -0.1047409, 0.6693002, -0.3087695, 0.8734686, -0.9782095, 0.9780697

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.5099635
time: 0.58 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635
time: 0.62 seconds

## Summary of splitting at layer (split count: 47)
- Time for IS candidates: 5.72 seconds
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 48, time: 5.72
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 48, time: 5.72
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 48, time: 5.72
Output dim: 0, lower bound: -0.5494671, upper bound: 0.5099635
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 48, time: 5.72
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0862837, 0.4352814, -0.0208821, 0.5835277, -0.4972440, 0.4561635
1: 0.0415683, 0.5686141, -0.0742011, 0.8010436, -0.7594753, 0.6428152
2: -0.0245677, 0.5098621, -0.1870105, 0.6244919, -0.6490597, 0.6968727
3: -0.0975448, 0.5112243, -0.2587125, 0.7709957, -0.8685405, 0.7699367
4: -0.1047409, 0.6693002, -0.3087695, 0.8734686, -0.9782095, 0.9780697

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5460619, upper bound: 0.5060034
time: 0.58 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.4901376
time: 0.58 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635
time: 0.59 seconds

## Summary of splitting at layer (split count: 48)
- Time for IS candidates: 6.95 seconds
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 49, time: 6.95
Output dim: 0, lower bound: -0.5664861, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 49, time: 6.95
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0903358, 0.4177946, -0.0208821, 0.5835277, -0.4931918, 0.4386767
1: 0.0493972, 0.5291730, -0.0742011, 0.8010436, -0.7516463, 0.6033741
2: -0.0079007, 0.5030103, -0.1870105, 0.6244919, -0.6323926, 0.6900208
3: -0.0871024, 0.4797198, -0.2587125, 0.7709957, -0.8580981, 0.7384322
4: -0.0905757, 0.6263897, -0.3087695, 0.8734686, -0.9640443, 0.9351592

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
time: 0.61 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
time: 0.60 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0862837, 0.4352814, -0.0208821, 0.5835277, -0.4972440, 0.4561635
1: 0.0415683, 0.5686141, -0.0742011, 0.8010436, -0.7594753, 0.6428152
2: -0.0245677, 0.5098621, -0.1870105, 0.6244919, -0.6490597, 0.6968727
3: -0.0975448, 0.5112243, -0.2587125, 0.7709957, -0.8685405, 0.7699367
4: -0.1047409, 0.6693002, -0.3087695, 0.8734686, -0.9782095, 0.9780697

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.5099635
time: 0.58 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635
time: 0.63 seconds

## Summary of splitting at layer (split count: 49)
- Time for IS candidates: 5.82 seconds
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 50, time: 5.82
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 50, time: 5.82
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 50, time: 5.82
Output dim: 0, lower bound: -0.5494671, upper bound: 0.5099635
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 50, time: 5.82
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0862837, 0.4352814, -0.0208821, 0.5835277, -0.4972440, 0.4561635
1: 0.0415683, 0.5686141, -0.0742011, 0.8010436, -0.7594753, 0.6428152
2: -0.0245677, 0.5098621, -0.1870105, 0.6244919, -0.6490597, 0.6968727
3: -0.0975448, 0.5112243, -0.2587125, 0.7709957, -0.8685405, 0.7699367
4: -0.1047409, 0.6693002, -0.3087695, 0.8734686, -0.9782095, 0.9780697

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5460619, upper bound: 0.5060034
time: 0.58 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.4901376
time: 0.59 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635
time: 0.60 seconds

## Summary of splitting at layer (split count: 50)
- Time for IS candidates: 7.01 seconds
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 51, time: 7.01
Output dim: 0, lower bound: -0.5664861, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 51, time: 7.01
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0903358, 0.4177946, -0.0208821, 0.5835277, -0.4931918, 0.4386767
1: 0.0493972, 0.5291730, -0.0742011, 0.8010436, -0.7516463, 0.6033741
2: -0.0079007, 0.5030103, -0.1870105, 0.6244919, -0.6323926, 0.6900208
3: -0.0871024, 0.4797198, -0.2587125, 0.7709957, -0.8580981, 0.7384322
4: -0.0905757, 0.6263897, -0.3087695, 0.8734686, -0.9640443, 0.9351592

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
time: 0.63 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
time: 0.68 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0862837, 0.4352814, -0.0208821, 0.5835277, -0.4972440, 0.4561635
1: 0.0415683, 0.5686141, -0.0742011, 0.8010436, -0.7594753, 0.6428152
2: -0.0245677, 0.5098621, -0.1870105, 0.6244919, -0.6490597, 0.6968727
3: -0.0975448, 0.5112243, -0.2587125, 0.7709957, -0.8685405, 0.7699367
4: -0.1047409, 0.6693002, -0.3087695, 0.8734686, -0.9782095, 0.9780697

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.5099635
time: 0.59 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635
time: 0.66 seconds

## Summary of splitting at layer (split count: 51)
- Time for IS candidates: 5.90 seconds
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 52, time: 5.90
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 52, time: 5.90
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 52, time: 5.90
Output dim: 0, lower bound: -0.5494671, upper bound: 0.5099635
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 52, time: 5.90
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0862837, 0.4352814, -0.0208821, 0.5835277, -0.4972440, 0.4561635
1: 0.0415683, 0.5686141, -0.0742011, 0.8010436, -0.7594753, 0.6428152
2: -0.0245677, 0.5098621, -0.1870105, 0.6244919, -0.6490597, 0.6968727
3: -0.0975448, 0.5112243, -0.2587125, 0.7709957, -0.8685405, 0.7699367
4: -0.1047409, 0.6693002, -0.3087695, 0.8734686, -0.9782095, 0.9780697

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5460619, upper bound: 0.5060034
time: 0.60 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.4901376
time: 0.60 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635
time: 0.61 seconds

## Summary of splitting at layer (split count: 52)
- Time for IS candidates: 7.12 seconds
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 53, time: 7.12
Output dim: 0, lower bound: -0.5664861, upper bound: 0.4901376
IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 53, time: 7.12
Output dim: 0, lower bound: -0.5664861, upper bound: 0.5099635

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0903358, 0.4177946, -0.0208821, 0.5835277, -0.4931918, 0.4386767
1: 0.0493972, 0.5291730, -0.0742011, 0.8010436, -0.7516463, 0.6033741
2: -0.0079007, 0.5030103, -0.1870105, 0.6244919, -0.6323926, 0.6900208
3: -0.0871024, 0.4797198, -0.2587125, 0.7709957, -0.8580981, 0.7384322
4: -0.0905757, 0.6263897, -0.3087695, 0.8734686, -0.9640443, 0.9351592

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
time: 0.59 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494671, upper bound: 0.4901376
time: 0.68 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0862837, 0.4352814, -0.0208821, 0.5835277, -0.4972440, 0.4561635
1: 0.0415683, 0.5686141, -0.0742011, 0.8010436, -0.7594753, 0.6428152
2: -0.0245677, 0.5098621, -0.1870105, 0.6244919, -0.6490597, 0.6968727
3: -0.0975448, 0.5112243, -0.2587125, 0.7709957, -0.8685405, 0.7699367
4: -0.1047409, 0.6693002, -0.3087695, 0.8734686, -0.9782095, 0.9780697

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 2): status=Status.UNKNOWN, low=0.0170836, high=0.0376754, mid=0.0376754, abs_max=0.6789970397949219
rel_dist={0: [-0.5783685500496472, 0.5783685500496476]}

## Binary search (step 3) starts
Candidate diff: 0.0273795


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5698305, upper bound: 0.5543882
time: 0.35 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5698305, upper bound: 0.5698305
time: 0.35 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.85 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.85
Output dim: 0, lower bound: -0.5698305, upper bound: 0.5543882
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.85
Output dim: 0, lower bound: -0.5698305, upper bound: 0.5698305

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.0674889, 0.4535027, -0.0115869, 0.5438887, -0.4735775, 0.4650896
1: 0.0201817, 0.5976362, -0.0674711, 0.7152010, -0.6950194, 0.6651073
2: -0.0460193, 0.5242411, -0.1453133, 0.6104485, -0.6564679, 0.6695544
3: -0.1192396, 0.5392005, -0.2312443, 0.6917207, -0.8109602, 0.7704449
4: -0.1227688, 0.6923177, -0.2496539, 0.7962284, -0.9189972, 0.9419717

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5623687, upper bound: 0.5520239
time: 0.38 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5677983, upper bound: 0.5529035
time: 0.38 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0106674, 0.5881721, -0.0250615, 0.6196039, -0.6302713, 0.6132336
1: -0.0680232, 0.7645921, -0.0850959, 0.8106070, -0.8786302, 0.8496880
2: -0.1492940, 0.6635130, -0.1746483, 0.6911386, -0.8404326, 0.8381613
3: -0.2353030, 0.7381678, -0.2637508, 0.7846647, -1.0199678, 1.0019186
4: -0.2566378, 0.8659467, -0.2888210, 0.9125841, -1.1692219, 1.1547676

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5543882, upper bound: 0.5698305
time: 0.40 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5543882, upper bound: 0.5698305
time: 0.33 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.43 seconds
IS_A1_A1, status: Status.VERIFIED, split count: 2, time: 2.43
Output dim: 0, lower bound: -0.5623687, upper bound: 0.5520239
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 2.43
Output dim: 0, lower bound: -0.5677983, upper bound: 0.5529035
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.43
Output dim: 0, lower bound: -0.5543882, upper bound: 0.5698305
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.43
Output dim: 0, lower bound: -0.5543882, upper bound: 0.5698305

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: 0.0789177, 0.4454548, -0.0115869, 0.5438887, -0.4649709, 0.4570417
1: 0.0311321, 0.5851125, -0.0674711, 0.7152010, -0.6840689, 0.6525837
2: -0.0367235, 0.5172547, -0.1453133, 0.6104485, -0.6471720, 0.6625680
3: -0.1111584, 0.5267107, -0.2312443, 0.6917207, -0.8028790, 0.7579550
4: -0.1149590, 0.6824017, -0.2496539, 0.7962284, -0.9111874, 0.9320556

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5614165, upper bound: 0.5450628
time: 0.34 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5614165, upper bound: 0.5529035
time: 0.33 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0106674, 0.5881721, 0.0688529, 0.4521295, -0.4627969, 0.5193192
1: -0.0680232, 0.7645921, 0.0226218, 0.5953072, -0.6633304, 0.7419704
2: -0.1492940, 0.6635130, -0.0440662, 0.5232872, -0.6725812, 0.7075792
3: -0.2353030, 0.7381678, -0.1174979, 0.5370846, -0.7723876, 0.8556657
4: -0.2566378, 0.8659467, -0.1212244, 0.6906068, -0.9472446, 0.9871711

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5520239, upper bound: 0.5593550
time: 0.35 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5529035, upper bound: 0.5677982
time: 0.33 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0106674, 0.5881721, -0.0106674, 0.5881721, -0.5988396, 0.5988396
1: -0.0680232, 0.7645921, -0.0680232, 0.7645921, -0.8326153, 0.8326153
2: -0.1492940, 0.6635130, -0.1492940, 0.6635130, -0.8128070, 0.8128070
3: -0.2353030, 0.7381678, -0.2353030, 0.7381678, -0.9734708, 0.9734708
4: -0.2566378, 0.8659467, -0.2566378, 0.8659467, -1.1225845, 1.1225845

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5520239, upper bound: 0.5593550
time: 0.35 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5529035, upper bound: 0.5677983
time: 0.36 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.39 seconds
IS_A1_A2_B1, status: Status.VERIFIED, split count: 3, time: 2.39
Output dim: 0, lower bound: -0.5614165, upper bound: 0.5450628
IS_A1_A2_B2, status: Status.VERIFIED, split count: 3, time: 2.39
Output dim: 0, lower bound: -0.5614165, upper bound: 0.5529035
IS_A2_B1_B1, status: Status.VERIFIED, split count: 3, time: 2.39
Output dim: 0, lower bound: -0.5520239, upper bound: 0.5593550
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 2.39
Output dim: 0, lower bound: -0.5529035, upper bound: 0.5677982
IS_A2_B2_B1, status: Status.VERIFIED, split count: 3, time: 2.39
Output dim: 0, lower bound: -0.5520239, upper bound: 0.5593550
IS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 2.39
Output dim: 0, lower bound: -0.5529035, upper bound: 0.5677983

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0106674, 0.5881721, 0.0797181, 0.4446977, -0.4553652, 0.5084540
1: -0.0680232, 0.7645921, 0.0318470, 0.5838390, -0.6518622, 0.7327451
2: -0.1492940, 0.6635130, -0.0357902, 0.5167565, -0.6660505, 0.6993032
3: -0.2353030, 0.7381678, -0.1101345, 0.5255548, -0.7608578, 0.8483022
4: -0.2566378, 0.8659467, -0.1140501, 0.6814760, -0.9381138, 0.9799968

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5450628, upper bound: 0.5614164
time: 0.37 seconds

## Relational analysis of IS_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5450628, upper bound: 0.5677983
time: 0.38 seconds

## BFS IS instance: IS_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0106674, 0.5881721, 0.0083361, 0.5719668, -0.5826342, 0.5798361
1: -0.0680232, 0.7645921, -0.0506135, 0.7433118, -0.8113350, 0.8152056
2: -0.1492940, 0.6635130, -0.1297052, 0.6455543, -0.7948483, 0.7932182
3: -0.2353030, 0.7381678, -0.2118729, 0.7126579, -0.9479610, 0.9500406
4: -0.2566378, 0.8659467, -0.2322460, 0.8428500, -1.0994878, 1.0981927

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_B2_A1

### Relational analysis result of IS_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5450628, upper bound: 0.5588391
time: 0.41 seconds

## Relational analysis of IS_A2_B2_B2_A2

### Relational analysis result of IS_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5450628, upper bound: 0.5677984
time: 0.38 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.49 seconds
IS_A2_B1_B2_A1, status: Status.VERIFIED, split count: 4, time: 2.49
Output dim: 0, lower bound: -0.5450628, upper bound: 0.5614164
IS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 0, lower bound: -0.5450628, upper bound: 0.5677983
IS_A2_B2_B2_A1, status: Status.VERIFIED, split count: 4, time: 2.49
Output dim: 0, lower bound: -0.5450628, upper bound: 0.5588391
IS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 0, lower bound: -0.5450628, upper bound: 0.5677984

## BFS IS instance: IS_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0083361, 0.5719668, 0.0797181, 0.4446977, -0.4363617, 0.4922487
1: -0.0506135, 0.7433118, 0.0318470, 0.5838390, -0.6344525, 0.7114648
2: -0.1297052, 0.6455543, -0.0357902, 0.5167565, -0.6464617, 0.6813445
3: -0.2118729, 0.7126579, -0.1101345, 0.5255548, -0.7374277, 0.8227924
4: -0.2322460, 0.8428500, -0.1140501, 0.6814760, -0.9137220, 0.9569001

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 5

Time for candidate selection: 4.38 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_B2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5341666, upper bound: 0.5646247
time: 0.36 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5426809, upper bound: 0.5593611
time: 0.38 seconds

## BFS IS instance: IS_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0083361, 0.5719668, 0.0083361, 0.5719668, -0.5636307, 0.5636307
1: -0.0506135, 0.7433118, -0.0506135, 0.7433118, -0.7939253, 0.7939253
2: -0.1297052, 0.6455543, -0.1297052, 0.6455543, -0.7752595, 0.7752595
3: -0.2118729, 0.7126579, -0.2118729, 0.7126579, -0.9245308, 0.9245308
4: -0.2322460, 0.8428500, -0.2322460, 0.8428500, -1.0750960, 1.0750960

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 24

Time for candidate selection: 4.48 seconds

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A2_B2_B2_A2_A1

### Relational analysis result of IS_A2_B2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5451665, upper bound: 0.5456919
time: 0.39 seconds

## Relational analysis of IS_A2_B2_B2_A2_A2

### Relational analysis result of IS_A2_B2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5335130, upper bound: 0.5407584
time: 0.43 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 6.85 seconds
IS_A2_B1_B2_A2_B1, status: Status.VERIFIED, split count: 5, time: 6.85
Output dim: 0, lower bound: -0.5341666, upper bound: 0.5646247
IS_A2_B1_B2_A2_B2, status: Status.VERIFIED, split count: 5, time: 6.85
Output dim: 0, lower bound: -0.5426809, upper bound: 0.5593611
IS_A2_B2_B2_A2_A1, status: Status.VERIFIED, split count: 5, time: 6.85
Output dim: 0, lower bound: -0.5451665, upper bound: 0.5456919
IS_A2_B2_B2_A2_A2, status: Status.VERIFIED, split count: 5, time: 6.85
Output dim: 0, lower bound: -0.5335130, upper bound: 0.5407584
Binary search (step 3): status=Status.VERIFIED, low=0.0273795, high=0.0376754, mid=0.0273795, abs_max=0.6789970397949219
rel_dist={0: [-0.572134639844827, 0.572134639844827]}

## Binary search (step 4) starts
Candidate diff: 0.0325275


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5730735, upper bound: 0.5551566
time: 0.42 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5730735, upper bound: 0.5730734
time: 0.36 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.93 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.93
Output dim: 0, lower bound: -0.5730735, upper bound: 0.5551566
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.93
Output dim: 0, lower bound: -0.5730735, upper bound: 0.5730734

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.0674889, 0.4535027, -0.0158160, 0.5608138, -0.4921860, 0.4693187
1: 0.0201817, 0.5976362, -0.0730230, 0.7377625, -0.7175809, 0.6706591
2: -0.0460193, 0.5242411, -0.1538767, 0.6274619, -0.6734812, 0.6781178
3: -0.1192396, 0.5392005, -0.2403692, 0.7146805, -0.8339201, 0.7795697
4: -0.1227688, 0.6923177, -0.2618473, 0.8219340, -0.9447027, 0.9541650

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5654509, upper bound: 0.5526722
time: 0.37 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5710985, upper bound: 0.5535342
time: 0.39 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0106674, 0.5881721, -0.0278299, 0.6251235, -0.6357909, 0.6160020
1: -0.0680232, 0.7645921, -0.0884378, 0.8183228, -0.8863460, 0.8530299
2: -0.1492940, 0.6635130, -0.1792251, 0.6961920, -0.8454860, 0.8427381
3: -0.2353030, 0.7381678, -0.2690502, 0.7929793, -1.0282824, 1.0072180
4: -0.2566378, 0.8659467, -0.2950232, 0.9207141, -1.1773520, 1.1609699

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5551566, upper bound: 0.5730735
time: 0.36 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5551566, upper bound: 0.5730735
time: 0.34 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.35 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 2.35
Output dim: 0, lower bound: -0.5654509, upper bound: 0.5526722
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 2.35
Output dim: 0, lower bound: -0.5710985, upper bound: 0.5535342
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.35
Output dim: 0, lower bound: -0.5551566, upper bound: 0.5730735
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.35
Output dim: 0, lower bound: -0.5551566, upper bound: 0.5730735

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: 0.0667267, 0.5276489, -0.0100148, 0.5494656, -0.4813285, 0.5376637
1: 0.0209925, 0.7202063, -0.0660595, 0.7213588, -0.7003663, 0.7862658
2: -0.0676523, 0.5800259, -0.1442511, 0.6165913, -0.6842436, 0.7242770
3: -0.1276886, 0.6454976, -0.2297661, 0.6971400, -0.8248287, 0.8752638
4: -0.1461909, 0.8046894, -0.2490106, 0.8039458, -0.9501367, 1.0537000

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5626389, upper bound: 0.5450628
time: 0.33 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5626389, upper bound: 0.5526722
time: 0.34 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: 0.0789177, 0.4454548, -0.0158160, 0.5608138, -0.4818961, 0.4612708
1: 0.0311321, 0.5851125, -0.0730230, 0.7377625, -0.7066304, 0.6581355
2: -0.0367235, 0.5172547, -0.1538767, 0.6274619, -0.6641853, 0.6711314
3: -0.1111584, 0.5267107, -0.2403692, 0.7146805, -0.8258389, 0.7670799
4: -0.1149590, 0.6824017, -0.2618473, 0.8219340, -0.9368930, 0.9442489

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5407517, upper bound: 0.5135096
time: 0.41 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5709419, upper bound: 0.5533864
time: 0.38 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0106674, 0.5881721, 0.0674889, 0.4535027, -0.4641702, 0.5202804
1: -0.0680232, 0.7645921, 0.0201817, 0.5976362, -0.6656594, 0.7444105
2: -0.1492940, 0.6635130, -0.0460193, 0.5242411, -0.6735351, 0.7095323
3: -0.2353030, 0.7381678, -0.1192396, 0.5392005, -0.7745036, 0.8574073
4: -0.2566378, 0.8659467, -0.1227688, 0.6923177, -0.9489555, 0.9887154

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5526722, upper bound: 0.5598309
time: 0.34 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5535342, upper bound: 0.5710984
time: 0.37 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0106674, 0.5881721, -0.0106674, 0.5881721, -0.5988396, 0.5988396
1: -0.0680232, 0.7645921, -0.0680232, 0.7645921, -0.8326153, 0.8326153
2: -0.1492940, 0.6635130, -0.1492940, 0.6635130, -0.8128070, 0.8128070
3: -0.2353030, 0.7381678, -0.2353030, 0.7381678, -0.9734708, 0.9734708
4: -0.2566378, 0.8659467, -0.2566378, 0.8659467, -1.1225845, 1.1225845

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5450628, upper bound: 0.5602331
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5535341, upper bound: 0.5711055
time: 0.40 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.46 seconds
IS_A1_A1_B1, status: Status.VERIFIED, split count: 3, time: 2.46
Output dim: 0, lower bound: -0.5626389, upper bound: 0.5450628
IS_A1_A1_B2, status: Status.VERIFIED, split count: 3, time: 2.46
Output dim: 0, lower bound: -0.5626389, upper bound: 0.5526722
IS_A1_A2_B1, status: Status.VERIFIED, split count: 3, time: 2.46
Output dim: 0, lower bound: -0.5407517, upper bound: 0.5135096
IS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 0, lower bound: -0.5709419, upper bound: 0.5533864
IS_A2_B1_B1, status: Status.VERIFIED, split count: 3, time: 2.46
Output dim: 0, lower bound: -0.5526722, upper bound: 0.5598309
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 0, lower bound: -0.5535342, upper bound: 0.5710984
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 2.46
Output dim: 0, lower bound: -0.5450628, upper bound: 0.5602331
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 0, lower bound: -0.5535341, upper bound: 0.5711055

## BFS IS instance: IS_A1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0789177, 0.4454548, -0.0148098, 0.5573949, -0.4784772, 0.4602647
1: 0.0311321, 0.5851125, -0.0717919, 0.7332022, -0.7020701, 0.6569044
2: -0.0367235, 0.5172547, -0.1519363, 0.6238335, -0.6605570, 0.6691910
3: -0.1111584, 0.5267107, -0.2381972, 0.7099926, -0.8211510, 0.7649079
4: -0.1149590, 0.6824017, -0.2590095, 0.8162601, -0.9312191, 0.9414111

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A2_B2_B1

### Relational analysis result of IS_A1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5632969, upper bound: 0.5449055
time: 0.37 seconds

## Relational analysis of IS_A1_A2_B2_B2

### Relational analysis result of IS_A1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5632969, upper bound: 0.5451149
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0106674, 0.5881721, 0.0789177, 0.4454548, -0.4561223, 0.5092544
1: -0.0680232, 0.7645921, 0.0311321, 0.5851125, -0.6531357, 0.7334600
2: -0.1492940, 0.6635130, -0.0367235, 0.5172547, -0.6665487, 0.7002365
3: -0.2353030, 0.7381678, -0.1111584, 0.5267107, -0.7620137, 0.8493261
4: -0.2566378, 0.8659467, -0.1149590, 0.6824017, -0.9390395, 0.9809057

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5450628, upper bound: 0.5634086
time: 0.36 seconds

## Relational analysis of IS_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5450628, upper bound: 0.5710985
time: 0.35 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0083361, 0.5719668, -0.0106674, 0.5881721, -0.5798361, 0.5826342
1: -0.0506135, 0.7433118, -0.0680232, 0.7645921, -0.8152056, 0.8113350
2: -0.1297052, 0.6455543, -0.1492940, 0.6635130, -0.7932182, 0.7948483
3: -0.2118729, 0.7126579, -0.2353030, 0.7381678, -0.9500406, 0.9479610
4: -0.2322460, 0.8428500, -0.2566378, 0.8659467, -1.0981927, 1.0994878

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5528364, upper bound: 0.5598309
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5526722, upper bound: 0.5711054
time: 0.45 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.56 seconds
IS_A1_A2_B2_B1, status: Status.VERIFIED, split count: 4, time: 2.56
Output dim: 0, lower bound: -0.5632969, upper bound: 0.5449055
IS_A1_A2_B2_B2, status: Status.VERIFIED, split count: 4, time: 2.56
Output dim: 0, lower bound: -0.5632969, upper bound: 0.5451149
IS_A2_B1_B2_A1, status: Status.VERIFIED, split count: 4, time: 2.56
Output dim: 0, lower bound: -0.5450628, upper bound: 0.5634086
IS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.56
Output dim: 0, lower bound: -0.5450628, upper bound: 0.5710985
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 2.56
Output dim: 0, lower bound: -0.5528364, upper bound: 0.5598309
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.56
Output dim: 0, lower bound: -0.5526722, upper bound: 0.5711054

## BFS IS instance: IS_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0083361, 0.5719668, 0.0789177, 0.4454548, -0.4371188, 0.4930490
1: -0.0506135, 0.7433118, 0.0311321, 0.5851125, -0.6357260, 0.7121797
2: -0.1297052, 0.6455543, -0.0367235, 0.5172547, -0.6469599, 0.6822778
3: -0.2118729, 0.7126579, -0.1111584, 0.5267107, -0.7385836, 0.8238163
4: -0.2322460, 0.8428500, -0.1149590, 0.6824017, -0.9146477, 0.9578090

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 5

Time for candidate selection: 4.48 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_B2_A2_A1

### Relational analysis result of IS_A2_B1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5443710, upper bound: 0.5493898
time: 0.39 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2

### Relational analysis result of IS_A2_B1_B2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5325719, upper bound: 0.5508181
time: 0.38 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0083361, 0.5719668, 0.0083361, 0.5719668, -0.5636307, 0.5636307
1: -0.0506135, 0.7433118, -0.0506135, 0.7433118, -0.7939253, 0.7939253
2: -0.1297052, 0.6455543, -0.1297052, 0.6455543, -0.7752595, 0.7752595
3: -0.2118729, 0.7126579, -0.2118729, 0.7126579, -0.9245308, 0.9245308
4: -0.2322460, 0.8428500, -0.2322460, 0.8428500, -1.0750960, 1.0750960

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B2_B1

### Relational analysis result of IS_A2_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5150354, upper bound: 0.5446060
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24

Time for candidate selection: 4.98 seconds

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5512237, upper bound: 0.5351808
time: 0.44 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5434514, upper bound: 0.5631170
time: 0.40 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 7.39 seconds
IS_A2_B1_B2_A2_A1, status: Status.VERIFIED, split count: 5, time: 7.39
Output dim: 0, lower bound: -0.5443710, upper bound: 0.5493898
IS_A2_B1_B2_A2_A2, status: Status.VERIFIED, split count: 5, time: 7.39
Output dim: 0, lower bound: -0.5325719, upper bound: 0.5508181
IS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 7.39
Output dim: 0, lower bound: -0.5512237, upper bound: 0.5351808
IS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 7.39
Output dim: 0, lower bound: -0.5434514, upper bound: 0.5631170
Binary search (step 4): status=Status.VERIFIED, low=0.0325275, high=0.0376754, mid=0.0325275, abs_max=0.6789970397949219
rel_dist={0: [-0.5753197067916472, 0.5753197067916478]}

## Binary search (step 5) starts
Candidate diff: 0.0351014


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5745144, upper bound: 0.5555408
time: 0.43 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5745144, upper bound: 0.5745144
time: 0.41 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.01 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.01
Output dim: 0, lower bound: -0.5745144, upper bound: 0.5555408
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.01
Output dim: 0, lower bound: -0.5745144, upper bound: 0.5745144

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.0674889, 0.4535027, -0.0178018, 0.5690731, -0.5013093, 0.4713045
1: 0.0201817, 0.5976362, -0.0756766, 0.7484761, -0.7282945, 0.6733127
2: -0.0460193, 0.5242411, -0.1580749, 0.6359100, -0.6819293, 0.6823159
3: -0.1192396, 0.5392005, -0.2447903, 0.7256426, -0.8448821, 0.7839909
4: -0.1227688, 0.6923177, -0.2674758, 0.8345305, -0.9572992, 0.9597936

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5669733, upper bound: 0.5529963
time: 0.35 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5726102, upper bound: 0.5538496
time: 0.35 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0106674, 0.5881721, -0.0292937, 0.6279622, -0.6386296, 0.6174658
1: -0.0680232, 0.7645921, -0.0902184, 0.8222815, -0.8903047, 0.8548105
2: -0.1492940, 0.6635130, -0.1816057, 0.6988317, -0.8481257, 0.8451187
3: -0.2353030, 0.7381678, -0.2718480, 0.7973796, -1.0326827, 1.0100157
4: -0.2566378, 0.8659467, -0.2984229, 0.9248393, -1.1814771, 1.1643696

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5555408, upper bound: 0.5745144
time: 0.35 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5555408, upper bound: 0.5745144
time: 0.43 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.43 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 2.43
Output dim: 0, lower bound: -0.5669733, upper bound: 0.5529963
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 2.43
Output dim: 0, lower bound: -0.5726102, upper bound: 0.5538496
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.43
Output dim: 0, lower bound: -0.5555408, upper bound: 0.5745144
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.43
Output dim: 0, lower bound: -0.5555408, upper bound: 0.5745144

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: 0.0667267, 0.5276489, -0.0122707, 0.5578263, -0.4905184, 0.5399196
1: 0.0209925, 0.7202063, -0.0689045, 0.7326885, -0.7116960, 0.7891108
2: -0.0676523, 0.5800259, -0.1486746, 0.6250550, -0.6927073, 0.7287005
3: -0.1276886, 0.6454976, -0.2345743, 0.7087817, -0.8364704, 0.8800719
4: -0.1461909, 0.8046894, -0.2553780, 0.8168043, -0.9629952, 1.0600674

Time for backsubstitution: 1.50 seconds
Binary search (step 5): status=Status.UNKNOWN, low=0.0325275, high=0.0351014, mid=0.0351014, abs_max=0.6789970397949219
rel_dist={0: [-0.576847000484482, 0.5768470004844821]}

## Binary Search with IS_dual Result
status: Status.VERIFIED
Maximum delta epsilon: 0.032527469390636554
execution time: 1151.47 seconds
