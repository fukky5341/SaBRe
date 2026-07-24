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
execution time: IAR + LP analysis = 1.63 + 1.03 = 2.65 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.5996373, upper bound: 0.5996373


# Binary Search by BASE starts (time budget: 1197.35 seconds, max iter: 100)

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
Binary search time: 48.09 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.017083602027241795


# Individual Split (IS_dual_ind) starts
Time budget: 1149.26 seconds

## Binary search (step 0) starts
Candidate diff: 0.0994509


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5950600, upper bound: 0.5628750
time: 0.33 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5885052, upper bound: 0.5885052
time: 0.38 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.86 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.86
Output dim: 0, lower bound: -0.5950600, upper bound: 0.5628750
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.86
Output dim: 0, lower bound: -0.5885052, upper bound: 0.5885052

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.0674889, 0.4535027, -0.0365533, 0.6424438, -0.5749549, 0.4900560
1: 0.0201817, 0.5976362, -0.0992057, 0.8423603, -0.8221787, 0.6968419
2: -0.0460193, 0.5242411, -0.1934414, 0.7124153, -0.7584347, 0.7176825
3: -0.1192396, 0.5392005, -0.2859350, 0.8194001, -0.9386396, 0.8251356
4: -0.1227688, 0.6923177, -0.3152393, 0.9457331, -1.0685018, 1.0075570

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5628750, upper bound: 0.5628750
time: 0.35 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5628750, upper bound: 0.5628750
time: 0.37 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0106674, 0.5881721, -0.0365533, 0.6424438, -0.6531112, 0.6247254
1: -0.0680232, 0.7645921, -0.0992057, 0.8423603, -0.9103835, 0.8637978
2: -0.1492940, 0.6635130, -0.1934414, 0.7124153, -0.8617094, 0.8569544
3: -0.2353030, 0.7381678, -0.2859350, 0.8194001, -1.0547031, 1.0241028
4: -0.2566378, 0.8659467, -0.3152393, 0.9457331, -1.2023709, 1.1811860

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5628750, upper bound: 0.5885052
time: 0.36 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5628750, upper bound: 0.5885052
time: 0.37 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.36 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 2.36
Output dim: 0, lower bound: -0.5628750, upper bound: 0.5628750
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 2.36
Output dim: 0, lower bound: -0.5628750, upper bound: 0.5628750
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.36
Output dim: 0, lower bound: -0.5628750, upper bound: 0.5885052
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.36
Output dim: 0, lower bound: -0.5628750, upper bound: 0.5885052

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0106674, 0.5881721, 0.0674889, 0.4535027, -0.4641702, 0.5206832
1: -0.0680232, 0.7645921, 0.0201817, 0.5976362, -0.6656594, 0.7444105
2: -0.1492940, 0.6635130, -0.0460193, 0.5242411, -0.6735351, 0.7095323
3: -0.2353030, 0.7381678, -0.1192396, 0.5392005, -0.7745036, 0.8574073
4: -0.2566378, 0.8659467, -0.1227688, 0.6923177, -0.9489555, 0.9887154

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5450628, upper bound: 0.5691239
time: 0.34 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5615309, upper bound: 0.5871832
time: 0.43 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0106674, 0.5881721, -0.0106674, 0.5881721, -0.5988396, 0.5988396
1: -0.0680232, 0.7645921, -0.0680232, 0.7645921, -0.8326153, 0.8326153
2: -0.1492940, 0.6635130, -0.1492940, 0.6635130, -0.8128070, 0.8128070
3: -0.2353030, 0.7381678, -0.2353030, 0.7381678, -0.9734708, 0.9734708
4: -0.2566378, 0.8659467, -0.2566378, 0.8659467, -1.1225845, 1.1225845

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5450628, upper bound: 0.5691239
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5615309, upper bound: 0.5871832
time: 0.34 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.39 seconds
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.39
Output dim: 0, lower bound: -0.5450628, upper bound: 0.5691239
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.39
Output dim: 0, lower bound: -0.5615309, upper bound: 0.5871832
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.39
Output dim: 0, lower bound: -0.5450628, upper bound: 0.5691239
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.39
Output dim: 0, lower bound: -0.5615309, upper bound: 0.5871832

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0069308, 0.7141405, 0.0676701, 0.4530736, -0.4600044, 0.6464704
1: -0.0711197, 0.9631418, 0.0204517, 0.5969021, -0.6680217, 0.9426901
2: -0.1773483, 0.7663486, -0.0456374, 0.5239233, -0.7012716, 0.8119860
3: -0.2441539, 0.9109904, -0.1189269, 0.5385518, -0.7827057, 1.0299172
4: -0.2841340, 1.0538890, -0.1224202, 0.6917305, -0.9758645, 1.1763092

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5450628, upper bound: 0.5820916
time: 0.34 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5450628, upper bound: 0.5821036
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0083361, 0.5719668, 0.0674889, 0.4535027, -0.4451666, 0.5044779
1: -0.0506135, 0.7433118, 0.0201817, 0.5976362, -0.6482497, 0.7231302
2: -0.1297052, 0.6455543, -0.0460193, 0.5242411, -0.6539463, 0.6915736
3: -0.2118729, 0.7126579, -0.1192396, 0.5392005, -0.7510734, 0.8318975
4: -0.2322460, 0.8428500, -0.1227688, 0.6923177, -0.9245638, 0.9656187

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5609737, upper bound: 0.5915545
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5609737, upper bound: 0.5920751
time: 0.34 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0069308, 0.7141405, -0.0101599, 0.5872346, -0.5941654, 0.7243004
1: -0.0711197, 0.9631418, -0.0674534, 0.7632908, -0.8344105, 1.0305952
2: -0.1773483, 0.7663486, -0.1485353, 0.6625104, -0.8398587, 0.9148839
3: -0.2441539, 0.9109904, -0.2343643, 0.7368354, -0.9809892, 1.1453546
4: -0.2841340, 1.0538890, -0.2556361, 0.8644702, -1.1486043, 1.3095251

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5450628, upper bound: 0.5532493
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5458074, upper bound: 0.5691239
time: 0.39 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0083361, 0.5719668, -0.0106674, 0.5881721, -0.5798361, 0.5826342
1: -0.0506135, 0.7433118, -0.0680232, 0.7645921, -0.8152056, 0.8113350
2: -0.1297052, 0.6455543, -0.1492940, 0.6635130, -0.7932182, 0.7948483
3: -0.2118729, 0.7126579, -0.2353030, 0.7381678, -0.9500406, 0.9479610
4: -0.2322460, 0.8428500, -0.2566378, 0.8659467, -1.0981927, 1.0994878

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5614976, upper bound: 0.5649611
time: 0.40 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5614976, upper bound: 0.5649611
time: 0.41 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.51 seconds
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.51
Output dim: 0, lower bound: -0.5450628, upper bound: 0.5820916
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.51
Output dim: 0, lower bound: -0.5450628, upper bound: 0.5821036
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.51
Output dim: 0, lower bound: -0.5609737, upper bound: 0.5915545
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.51
Output dim: 0, lower bound: -0.5609737, upper bound: 0.5920751
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.51
Output dim: 0, lower bound: -0.5450628, upper bound: 0.5532493
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.51
Output dim: 0, lower bound: -0.5458074, upper bound: 0.5691239
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 2.51
Output dim: 0, lower bound: -0.5614976, upper bound: 0.5649611
IS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 2.51
Output dim: 0, lower bound: -0.5614976, upper bound: 0.5649611

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0069308, 0.7141405, 0.0667267, 0.5276489, -0.5345798, 0.6474138
1: -0.0711197, 0.9631418, 0.0209925, 0.7202063, -0.7913260, 0.9421493
2: -0.1773483, 0.7663486, -0.0676523, 0.5800259, -0.7573742, 0.8340009
3: -0.2441539, 0.9109904, -0.1276886, 0.6454976, -0.8896515, 1.0386790
4: -0.2841340, 1.0538890, -0.1461909, 0.8046894, -1.0888234, 1.2000799

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4692780, upper bound: 0.5564279
time: 0.34 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4679420, upper bound: 0.5819449
time: 0.34 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24

Time for candidate selection: 2.97 seconds

### Candidate
type: A, layer: 3, pos: 39

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5443710, upper bound: 0.5600484
time: 0.34 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5325719, upper bound: 0.5729790
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0069308, 0.7141405, 0.0789177, 0.4454548, -0.4523857, 0.6352228
1: -0.0711197, 0.9631418, 0.0311321, 0.5851125, -0.6562322, 0.9320097
2: -0.1773483, 0.7663486, -0.0367235, 0.5172547, -0.6946030, 0.8030721
3: -0.2441539, 0.9109904, -0.1111584, 0.5267107, -0.7708645, 1.0221487
4: -0.2841340, 1.0538890, -0.1149590, 0.6824017, -0.9665357, 1.1688480

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4692780, upper bound: 0.5564279
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4679420, upper bound: 0.5819547
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24

Time for candidate selection: 3.03 seconds

### Candidate
type: A, layer: 3, pos: 39

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5443710, upper bound: 0.5600834
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5325719, upper bound: 0.5730140
time: 0.34 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0083361, 0.5719668, 0.0667267, 0.5276489, -0.5193129, 0.5052401
1: -0.0506135, 0.7433118, 0.0209925, 0.7202063, -0.7708198, 0.7223193
2: -0.1297052, 0.6455543, -0.0676523, 0.5800259, -0.7097311, 0.7132066
3: -0.2118729, 0.7126579, -0.1276886, 0.6454976, -0.8573705, 0.8403466
4: -0.2322460, 0.8428500, -0.1461909, 0.8046894, -1.0369354, 0.9890409

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4865432, upper bound: 0.5794595
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5608158, upper bound: 0.5915238
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0083361, 0.5719668, 0.0789177, 0.4454548, -0.4371188, 0.4930490
1: -0.0506135, 0.7433118, 0.0311321, 0.5851125, -0.6357260, 0.7121797
2: -0.1297052, 0.6455543, -0.0367235, 0.5172547, -0.6469599, 0.6822778
3: -0.2118729, 0.7126579, -0.1111584, 0.5267107, -0.7385836, 0.8238163
4: -0.2322460, 0.8428500, -0.1149590, 0.6824017, -0.9146477, 0.9578090

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4865434, upper bound: 0.5916821
time: 0.34 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5608159, upper bound: 0.5915238
time: 0.37 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0069308, 0.7141405, 0.0083361, 0.5719668, -0.5788976, 0.7058045
1: -0.0711197, 0.9631418, -0.0506135, 0.7433118, -0.8144315, 1.0137553
2: -0.1773483, 0.7663486, -0.1297052, 0.6455543, -0.8229026, 0.8960538
3: -0.2441539, 0.9109904, -0.2118729, 0.7126579, -0.9568118, 1.1228633
4: -0.2841340, 1.0538890, -0.2322460, 0.8428500, -1.1269840, 1.2861351

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24

Time for candidate selection: 2.31 seconds

### Candidate
type: A, layer: 3, pos: 39

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5451665, upper bound: 0.5488237
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5335130, upper bound: 0.5625546
time: 0.37 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 5.36 seconds
IS_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 5.36
Output dim: 0, lower bound: -0.5443710, upper bound: 0.5600484
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -0.5325719, upper bound: 0.5729790
IS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 5.36
Output dim: 0, lower bound: -0.5443710, upper bound: 0.5600834
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -0.5325719, upper bound: 0.5730140
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -0.4865432, upper bound: 0.5794595
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -0.5608158, upper bound: 0.5915238
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -0.4865434, upper bound: 0.5916821
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -0.5608159, upper bound: 0.5915238
IS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 5.36
Output dim: 0, lower bound: -0.5451665, upper bound: 0.5488237
IS_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 5.36
Output dim: 0, lower bound: -0.5335130, upper bound: 0.5625546

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0073434, 0.7021115, 0.0667267, 0.5276489, -0.5203055, 0.6353848
1: -0.0522960, 0.9448535, 0.0209925, 0.7202063, -0.7725023, 0.9238610
2: -0.1551384, 0.7545946, -0.0676523, 0.5800259, -0.7351643, 0.8222469
3: -0.2195477, 0.8881526, -0.1276886, 0.6454976, -0.8650453, 1.0158412
4: -0.2558787, 1.0356774, -0.1461909, 0.8046894, -1.0605681, 1.1818683

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.30 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5305236, upper bound: 0.5253708
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5281923, upper bound: 0.5617464
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0073434, 0.7021115, 0.0789177, 0.4454548, -0.4381114, 0.6231937
1: -0.0522960, 0.9448535, 0.0311321, 0.5851125, -0.6374085, 0.9137214
2: -0.1551384, 0.7545946, -0.0367235, 0.5172547, -0.6723931, 0.7913181
3: -0.2195477, 0.8881526, -0.1111584, 0.5267107, -0.7462584, 0.9993110
4: -0.2558787, 1.0356774, -0.1149590, 0.6824017, -0.9382803, 1.1506364

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.32 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5297427, upper bound: 0.5353883
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5282073, upper bound: 0.5602669
time: 0.34 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0116752, 0.6079841, 0.0672615, 0.5269604, -0.5386357, 0.5407226
1: -0.0747348, 0.8284329, 0.0219309, 0.7190909, -0.7938257, 0.8065019
2: -0.1899765, 0.6503597, -0.0668643, 0.5795007, -0.7694772, 0.7172240
3: -0.2612604, 0.7920903, -0.1268568, 0.6444501, -0.9057105, 0.9189471
4: -0.3060377, 0.9010377, -0.1454698, 0.8038490, -1.1098866, 1.0465075

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.37 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5772907
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4835004, upper bound: 0.5771162
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0100145, 0.5674599, 0.0667267, 0.5276489, -0.5176344, 0.5007333
1: -0.0486783, 0.7368349, 0.0209925, 0.7202063, -0.7688846, 0.7158424
2: -0.1268167, 0.6409237, -0.0676523, 0.5800259, -0.7068427, 0.7085760
3: -0.2082636, 0.7057808, -0.1276886, 0.6454976, -0.8537612, 0.8334695
4: -0.2279943, 0.8359103, -0.1461909, 0.8046894, -1.0326837, 0.9821012

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5589591, upper bound: 0.5778258
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5119762, upper bound: 0.5748654
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 3.14 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5568824, upper bound: 0.5282470
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5564721, upper bound: 0.5725318
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0116752, 0.6079841, 0.0793476, 0.4448406, -0.4565158, 0.5286365
1: -0.0747348, 0.8284329, 0.0316794, 0.5841250, -0.6588598, 0.7967535
2: -0.1899765, 0.6503597, -0.0360711, 0.5167693, -0.7067457, 0.6864308
3: -0.2612604, 0.7920903, -0.1104538, 0.5257305, -0.7869909, 0.9025441
4: -0.3060377, 0.9010377, -0.1143421, 0.6816227, -0.9876604, 1.0153798

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5266696, upper bound: 0.5802624
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5205424, upper bound: 0.5853410
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0100145, 0.5674599, 0.0789177, 0.4454548, -0.4354403, 0.4885422
1: -0.0486783, 0.7368349, 0.0311321, 0.5851125, -0.6337908, 0.7057028
2: -0.1268167, 0.6409237, -0.0367235, 0.5172547, -0.6440715, 0.6776472
3: -0.2082636, 0.7057808, -0.1111584, 0.5267107, -0.7349743, 0.8169392
4: -0.2279943, 0.8359103, -0.1149590, 0.6824017, -0.9103960, 0.9508693

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5607108, upper bound: 0.5806441
time: 0.44 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5506119, upper bound: 0.5853645
time: 0.42 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 2.57 seconds
IS_A2_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.57
Output dim: 0, lower bound: -0.5305236, upper bound: 0.5253708
IS_A2_B1_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 2.57
Output dim: 0, lower bound: -0.5281923, upper bound: 0.5617464
IS_A2_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.57
Output dim: 0, lower bound: -0.5297427, upper bound: 0.5353883
IS_A2_B1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 2.57
Output dim: 0, lower bound: -0.5282073, upper bound: 0.5602669
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.57
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5772907
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.57
Output dim: 0, lower bound: -0.4835004, upper bound: 0.5771162
IS_A2_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.57
Output dim: 0, lower bound: -0.5568824, upper bound: 0.5282470
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.57
Output dim: 0, lower bound: -0.5564721, upper bound: 0.5725318
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.57
Output dim: 0, lower bound: -0.5266696, upper bound: 0.5802624
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.57
Output dim: 0, lower bound: -0.5205424, upper bound: 0.5853410
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.57
Output dim: 0, lower bound: -0.5607108, upper bound: 0.5806441
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.57
Output dim: 0, lower bound: -0.5506119, upper bound: 0.5853645

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0116752, 0.6079841, 0.0774577, 0.4984086, -0.5100838, 0.5305264
1: -0.0747348, 0.8284329, 0.0378881, 0.6620049, -0.7367398, 0.7905448
2: -0.1899765, 0.6503597, -0.0404382, 0.5646217, -0.7545981, 0.6907979
3: -0.2612604, 0.7920903, -0.1053302, 0.5949185, -0.8561789, 0.8974205
4: -0.3060377, 0.9010377, -0.1228297, 0.7481163, -1.0541539, 1.0238674

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24

Time for candidate selection: 2.08 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5694776
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5771162
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0116752, 0.6079841, 0.0739317, 0.5186416, -0.5303168, 0.5340524
1: -0.0747348, 0.8284329, 0.0327510, 0.7059036, -0.7806385, 0.7956819
2: -0.1899765, 0.6503597, -0.0554504, 0.5733088, -0.7632853, 0.7058102
3: -0.2612604, 0.7920903, -0.1144004, 0.6320976, -0.8933580, 0.9064907
4: -0.3060377, 0.9010377, -0.1361425, 0.7930427, -1.0990803, 1.0371802

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24

Time for candidate selection: 2.03 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4835004, upper bound: 0.5694776
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4835004, upper bound: 0.5771162
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0100145, 0.5674599, 0.0677700, 0.5231507, -0.5131362, 0.4996899
1: -0.0486783, 0.7368349, 0.0225216, 0.7127589, -0.7614372, 0.7143133
2: -0.1268167, 0.6409237, -0.0648993, 0.5771931, -0.7040099, 0.7058230
3: -0.2082636, 0.7057808, -0.1257259, 0.6387789, -0.8470425, 0.8315067
4: -0.2279943, 0.8359103, -0.1438800, 0.7990351, -1.0270294, 0.9797903

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5308305, upper bound: 0.5720046
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5201014, upper bound: 0.5724502
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 36

Time for candidate selection: 2.81 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5552753, upper bound: 0.5689627
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5407232, upper bound: 0.5689627
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0116752, 0.6079841, 0.0894709, 0.4175571, -0.4292324, 0.5185132
1: -0.0747348, 0.8284329, 0.0457507, 0.5343553, -0.6090901, 0.7826821
2: -0.1899765, 0.6503597, -0.0096244, 0.5019960, -0.6919725, 0.6599841
3: -0.2612604, 0.7920903, -0.0870266, 0.4841275, -0.7453879, 0.8791169
4: -0.3060377, 0.9010377, -0.0887296, 0.6444156, -0.9504533, 0.9897673

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5065091, upper bound: 0.5802624
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5065091, upper bound: 0.5802624
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0116752, 0.6079841, 0.0805085, 0.4525800, -0.4642552, 0.5274756
1: -0.0747348, 0.8284329, 0.0280254, 0.5985373, -0.6732721, 0.8004075
2: -0.1899765, 0.6503597, -0.0423912, 0.5218675, -0.7118440, 0.6927509
3: -0.2612604, 0.7920903, -0.1158811, 0.5381635, -0.7994239, 0.9079714
4: -0.3060377, 0.9010377, -0.1197159, 0.6938162, -0.9998538, 1.0207536

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5065091, upper bound: 0.5853410
time: 0.44 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5065091, upper bound: 0.5853410
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0100145, 0.5674599, 0.0890627, 0.4180824, -0.4080678, 0.4783973
1: -0.0486783, 0.7368349, 0.0450087, 0.5352120, -0.5838903, 0.6918262
2: -0.1268167, 0.6409237, -0.0104511, 0.5023968, -0.6292135, 0.6513748
3: -0.2082636, 0.7057808, -0.0877064, 0.4849638, -0.6932274, 0.7934873
4: -0.2279943, 0.8359103, -0.0893122, 0.6450840, -0.8730783, 0.9252225

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5410317, upper bound: 0.5806441
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5410317, upper bound: 0.5806441
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0100145, 0.5674599, 0.0801258, 0.4531720, -0.4431575, 0.4873341
1: -0.0486783, 0.7368349, 0.0275328, 0.5994896, -0.6481679, 0.7093021
2: -0.1268167, 0.6409237, -0.0430036, 0.5223081, -0.6491249, 0.6839272
3: -0.2082636, 0.7057808, -0.1165140, 0.5390931, -0.7473567, 0.8222948
4: -0.2279943, 0.8359103, -0.1202720, 0.6945584, -0.9225527, 0.9561824

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5410318, upper bound: 0.5853645
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5410318, upper bound: 0.5853645
time: 0.43 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 2.53 seconds
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5694776
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5771162
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.4835004, upper bound: 0.5694776
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.4835004, upper bound: 0.5771162
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5552753, upper bound: 0.5689627
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5407232, upper bound: 0.5689627
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5065091, upper bound: 0.5802624
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5065091, upper bound: 0.5802624
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5065091, upper bound: 0.5853410
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5065091, upper bound: 0.5853410
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5410317, upper bound: 0.5806441
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5410317, upper bound: 0.5806441
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5410318, upper bound: 0.5853645
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5410318, upper bound: 0.5853645

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0186715, 0.5356178, 0.0774577, 0.4984086, -0.4797370, 0.4581601
1: -0.0405200, 0.7052470, 0.0378881, 0.6620049, -0.7025249, 0.6673590
2: -0.1335938, 0.5874144, -0.0404382, 0.5646217, -0.6982155, 0.6278526
3: -0.2105730, 0.6657733, -0.1053302, 0.5949185, -0.8054915, 0.7711036
4: -0.2271247, 0.7838589, -0.1228297, 0.7481163, -0.9752410, 0.9066886

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.41 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5739569
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5694776
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0054767, 0.5886192, 0.0774577, 0.4984086, -0.4929318, 0.5111616
1: -0.0523044, 0.7926955, 0.0378881, 0.6620049, -0.7143093, 0.7548075
2: -0.1597649, 0.6353322, -0.0404382, 0.5646217, -0.7243866, 0.6757704
3: -0.2289202, 0.7572249, -0.1053302, 0.5949185, -0.8238387, 0.8625551
4: -0.2726064, 0.8697696, -0.1228297, 0.7481163, -1.0207226, 0.9925992

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.40 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5772907
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5771162
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0186715, 0.5356178, 0.0739317, 0.5186416, -0.4999701, 0.4616861
1: -0.0405200, 0.7052470, 0.0327510, 0.7059036, -0.7464236, 0.6724961
2: -0.1335938, 0.5874144, -0.0554504, 0.5733088, -0.7069026, 0.6428648
3: -0.2105730, 0.6657733, -0.1144004, 0.6320976, -0.8426706, 0.7801738
4: -0.2271247, 0.7838589, -0.1361425, 0.7930427, -1.0201674, 0.9200014

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.39 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5694776
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5694776
time: 0.33 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0054767, 0.5886192, 0.0739317, 0.5186416, -0.5131649, 0.5146875
1: -0.0523044, 0.7926955, 0.0327510, 0.7059036, -0.7582080, 0.7599446
2: -0.1597649, 0.6353322, -0.0554504, 0.5733088, -0.7330738, 0.6907827
3: -0.2289202, 0.7572249, -0.1144004, 0.6320976, -0.8610178, 0.8716253
4: -0.2726064, 0.8697696, -0.1361425, 0.7930427, -1.0656490, 1.0059121

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.41 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5771162
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4835004, upper bound: 0.5771162
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0289738, 0.5272163, 0.0677700, 0.5231507, -0.4941769, 0.4594463
1: -0.0271752, 0.6625315, 0.0225216, 0.7127589, -0.7399341, 0.6400099
2: -0.0859941, 0.6136217, -0.0648993, 0.5771931, -0.6631873, 0.6785210
3: -0.1722872, 0.6395842, -0.1257259, 0.6387789, -0.8110661, 0.7653100
4: -0.1796645, 0.7690048, -0.1438800, 0.7990351, -0.9786996, 0.9128848

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5530363, upper bound: 0.5597668
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.90 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5552753, upper bound: 0.5261013
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5552753, upper bound: 0.5689627
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0166713, 0.5612808, 0.0677700, 0.5231507, -0.5064794, 0.4935108
1: -0.0441760, 0.7052452, 0.0225216, 0.7127589, -0.7569349, 0.6827236
2: -0.1037405, 0.6534984, -0.0648993, 0.5771931, -0.6809336, 0.7183977
3: -0.2023969, 0.6899778, -0.1257259, 0.6387789, -0.8411758, 0.8157037
4: -0.2070733, 0.8241372, -0.1438800, 0.7990351, -1.0061084, 0.9680172

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.53 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5407232, upper bound: 0.5261013
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5407232, upper bound: 0.5689627
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0047441, 0.5822396, 0.0894709, 0.4175571, -0.4128131, 0.4927688
1: -0.0574056, 0.7926347, 0.0457507, 0.5343553, -0.5917609, 0.7468840
2: -0.1635008, 0.6239193, -0.0096244, 0.5019960, -0.6654968, 0.6335437
3: -0.2339801, 0.7492464, -0.0870266, 0.4841275, -0.7181076, 0.8362730
4: -0.2717005, 0.8646826, -0.0887296, 0.6444156, -0.9161161, 0.9534122

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.05 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4938970, upper bound: 0.5792839
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5081566, upper bound: 0.5696791
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0097460, 0.6148499, 0.0894709, 0.4175571, -0.4273031, 0.5253791
1: -0.0750871, 0.8550988, 0.0457507, 0.5343553, -0.6094424, 0.8093481
2: -0.1931111, 0.6471984, -0.0096244, 0.5019960, -0.6951071, 0.6568228
3: -0.2629132, 0.8158054, -0.0870266, 0.4841275, -0.7470406, 0.9028320
4: -0.3095037, 0.9171438, -0.0887296, 0.6444156, -0.9539193, 1.0058734

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.08 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4938970, upper bound: 0.5792839
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5081566, upper bound: 0.5696791
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0047441, 0.5822396, 0.0805085, 0.4525800, -0.4478359, 0.5017312
1: -0.0574056, 0.7926347, 0.0280254, 0.5985373, -0.6559429, 0.7646093
2: -0.1635008, 0.6239193, -0.0423912, 0.5218675, -0.6853683, 0.6663105
3: -0.2339801, 0.7492464, -0.1158811, 0.5381635, -0.7721436, 0.8651274
4: -0.2717005, 0.8646826, -0.1197159, 0.6938162, -0.9655167, 0.9843985

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.04 seconds

### Candidate
type: B, layer: 3, pos: 41

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5065091, upper bound: 0.5853410
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4865263, upper bound: 0.5837921
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5008990, upper bound: 0.5770830
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0097460, 0.6148499, 0.0805085, 0.4525800, -0.4623259, 0.5343415
1: -0.0750871, 0.8550988, 0.0280254, 0.5985373, -0.6736243, 0.8270735
2: -0.1931111, 0.6471984, -0.0423912, 0.5218675, -0.7149786, 0.6895896
3: -0.2629132, 0.8158054, -0.1158811, 0.5381635, -0.8010767, 0.9316865
4: -0.3095037, 0.9171438, -0.1197159, 0.6938162, -1.0033199, 1.0368598

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.07 seconds

### Candidate
type: B, layer: 3, pos: 41

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5065090, upper bound: 0.5804924
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4865264, upper bound: 0.5794951
time: 0.45 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5008990, upper bound: 0.5702969
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0294979, 0.5303939, 0.0890627, 0.4180824, -0.3885845, 0.4413312
1: -0.0272393, 0.6770903, 0.0450087, 0.5352120, -0.5624513, 0.6320816
2: -0.0920497, 0.6105885, -0.0104511, 0.5023968, -0.5944465, 0.6210396
3: -0.1739870, 0.6456757, -0.0877064, 0.4849638, -0.6589507, 0.7333822
4: -0.1849493, 0.7805903, -0.0893122, 0.6450840, -0.8300333, 0.8699025

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.10 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5519187, upper bound: 0.5136782
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5503832, upper bound: 0.5629116
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0056896, 0.5840424, 0.0890627, 0.4180824, -0.4123927, 0.4949797
1: -0.0572673, 0.7667230, 0.0450087, 0.5352120, -0.5924793, 0.7217144
2: -0.1407367, 0.6509148, -0.0104511, 0.5023968, -0.6431335, 0.6613659
3: -0.2236565, 0.7371542, -0.0877064, 0.4849638, -0.7086203, 0.8248606
4: -0.2477118, 0.8606789, -0.0893122, 0.6450840, -0.8927958, 0.9499911

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.06 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5519187, upper bound: 0.5136782
time: 0.47 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5503832, upper bound: 0.5629116
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0294979, 0.5303939, 0.0801258, 0.4531720, -0.4236742, 0.4502681
1: -0.0272393, 0.6770903, 0.0275328, 0.5994896, -0.6267290, 0.6495575
2: -0.0920497, 0.6105885, -0.0430036, 0.5223081, -0.6143578, 0.6535921
3: -0.1739870, 0.6456757, -0.1165140, 0.5390931, -0.7130800, 0.7621897
4: -0.1849493, 0.7805903, -0.1202720, 0.6945584, -0.8795077, 0.9008623

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.12 seconds

### Candidate
type: B, layer: 3, pos: 41

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5410317, upper bound: 0.5853645
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5169797, upper bound: 0.5838156
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5313523, upper bound: 0.5771071
time: 0.45 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0056896, 0.5840424, 0.0801258, 0.4531720, -0.4474824, 0.5039166
1: -0.0572673, 0.7667230, 0.0275328, 0.5994896, -0.6567569, 0.7391902
2: -0.1407367, 0.6509148, -0.0430036, 0.5223081, -0.6630448, 0.6939183
3: -0.2236565, 0.7371542, -0.1165140, 0.5390931, -0.7627496, 0.8536682
4: -0.2477118, 0.8606789, -0.1202720, 0.6945584, -0.9422702, 0.9809510

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.08 seconds

### Candidate
type: B, layer: 3, pos: 41

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5410317, upper bound: 0.5806447
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5169797, upper bound: 0.5797515
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5313523, upper bound: 0.5704499
time: 0.40 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 5.37 seconds
IS_A2_B1_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.37
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5739569
IS_A2_B1_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.37
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5694776
IS_A2_B1_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.37
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5772907
IS_A2_B1_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.37
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5771162
IS_A2_B1_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.37
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5694776
IS_A2_B1_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.37
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5694776
IS_A2_B1_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.37
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5771162
IS_A2_B1_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.37
Output dim: 0, lower bound: -0.4835004, upper bound: 0.5771162
IS_A2_B1_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 5.37
Output dim: 0, lower bound: -0.5552753, upper bound: 0.5261013
IS_A2_B1_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.37
Output dim: 0, lower bound: -0.5552753, upper bound: 0.5689627
IS_A2_B1_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 5.37
Output dim: 0, lower bound: -0.5407232, upper bound: 0.5261013
IS_A2_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.37
Output dim: 0, lower bound: -0.5407232, upper bound: 0.5689627
IS_A2_B1_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.37
Output dim: 0, lower bound: -0.4938970, upper bound: 0.5792839
IS_A2_B1_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.37
Output dim: 0, lower bound: -0.5081566, upper bound: 0.5696791
IS_A2_B1_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.37
Output dim: 0, lower bound: -0.4938970, upper bound: 0.5792839
IS_A2_B1_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.37
Output dim: 0, lower bound: -0.5081566, upper bound: 0.5696791
IS_A2_B1_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.37
Output dim: 0, lower bound: -0.4865263, upper bound: 0.5837921
IS_A2_B1_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.37
Output dim: 0, lower bound: -0.5008990, upper bound: 0.5770830
IS_A2_B1_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.37
Output dim: 0, lower bound: -0.4865264, upper bound: 0.5794951
IS_A2_B1_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.37
Output dim: 0, lower bound: -0.5008990, upper bound: 0.5702969
IS_A2_B1_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 5.37
Output dim: 0, lower bound: -0.5519187, upper bound: 0.5136782
IS_A2_B1_A2_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 5.37
Output dim: 0, lower bound: -0.5503832, upper bound: 0.5629116
IS_A2_B1_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 5.37
Output dim: 0, lower bound: -0.5519187, upper bound: 0.5136782
IS_A2_B1_A2_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 5.37
Output dim: 0, lower bound: -0.5503832, upper bound: 0.5629116
IS_A2_B1_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.37
Output dim: 0, lower bound: -0.5169797, upper bound: 0.5838156
IS_A2_B1_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.37
Output dim: 0, lower bound: -0.5313523, upper bound: 0.5771071
IS_A2_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.37
Output dim: 0, lower bound: -0.5169797, upper bound: 0.5797515
IS_A2_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.37
Output dim: 0, lower bound: -0.5313523, upper bound: 0.5704499

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0186715, 0.5356178, 0.0774577, 0.4984086, -0.4797370, 0.4581601
1: -0.0405200, 0.7052470, 0.0378881, 0.6620049, -0.7025249, 0.6673590
2: -0.1335938, 0.5874144, -0.0404382, 0.5646217, -0.6982155, 0.6278526
3: -0.2105730, 0.6657733, -0.1053302, 0.5949185, -0.8054915, 0.7711036
4: -0.2271247, 0.7838589, -0.1228297, 0.7481163, -0.9752410, 0.9066886

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 36

Time for candidate selection: 2.04 seconds

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4662963, upper bound: 0.5675739
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4683341, upper bound: 0.5694776
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5694776
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0186715, 0.5356178, 0.0739940, 0.5183154, -0.4996439, 0.4616237
1: -0.0405200, 0.7052470, 0.0328778, 0.7053200, -0.7458400, 0.6723692
2: -0.1335938, 0.5874144, -0.0552860, 0.5731409, -0.7067347, 0.6427004
3: -0.2105730, 0.6657733, -0.1142519, 0.6315680, -0.8421409, 0.7800252
4: -0.2271247, 0.7838589, -0.1360635, 0.7927270, -1.0198517, 0.9199224

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 36

Time for candidate selection: 2.10 seconds

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4662963, upper bound: 0.5675739
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4683341, upper bound: 0.5694776
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5694776
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0054767, 0.5886192, 0.0774577, 0.4984086, -0.4929318, 0.5111616
1: -0.0523044, 0.7926955, 0.0378881, 0.6620049, -0.7143093, 0.7548075
2: -0.1597649, 0.6353322, -0.0404382, 0.5646217, -0.7243866, 0.6757704
3: -0.2289202, 0.7572249, -0.1053302, 0.5949185, -0.8238387, 0.8625551
4: -0.2726064, 0.8697696, -0.1228297, 0.7481163, -1.0207226, 0.9925992

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24

Time for candidate selection: 2.12 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5694776
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5771162
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0054767, 0.5886192, 0.0739940, 0.5183154, -0.5128387, 0.5146252
1: -0.0523044, 0.7926955, 0.0328778, 0.7053200, -0.7576244, 0.7598177
2: -0.1597649, 0.6353322, -0.0552860, 0.5731409, -0.7329058, 0.6906183
3: -0.2289202, 0.7572249, -0.1142519, 0.6315680, -0.8604882, 0.8714768
4: -0.2726064, 0.8697696, -0.1360635, 0.7927270, -1.0653334, 1.0058330

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24

Time for candidate selection: 2.16 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5694776
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5771162
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0186715, 0.5356178, 0.0774577, 0.4984086, -0.4797370, 0.4581601
1: -0.0405200, 0.7052470, 0.0378881, 0.6620049, -0.7025249, 0.6673590
2: -0.1335938, 0.5874144, -0.0404382, 0.5646217, -0.6982155, 0.6278526
3: -0.2105730, 0.6657733, -0.1053302, 0.5949185, -0.8054915, 0.7711036
4: -0.2271247, 0.7838589, -0.1228297, 0.7481163, -0.9752410, 0.9066886

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 36

Time for candidate selection: 2.12 seconds

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4662963, upper bound: 0.5675739
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4683341, upper bound: 0.5694776
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5694776
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0186715, 0.5356178, 0.0739317, 0.5186416, -0.4999701, 0.4616861
1: -0.0405200, 0.7052470, 0.0327510, 0.7059036, -0.7464236, 0.6724961
2: -0.1335938, 0.5874144, -0.0554504, 0.5733088, -0.7069026, 0.6428648
3: -0.2105730, 0.6657733, -0.1144004, 0.6320976, -0.8426706, 0.7801738
4: -0.2271247, 0.7838589, -0.1361425, 0.7930427, -1.0201674, 0.9200014

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 36

Time for candidate selection: 2.12 seconds

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4662963, upper bound: 0.5675739
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4854941, upper bound: 0.5694776
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4835004, upper bound: 0.5694776
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0054767, 0.5886192, 0.0774577, 0.4984086, -0.4929318, 0.5111616
1: -0.0523044, 0.7926955, 0.0378881, 0.6620049, -0.7143093, 0.7548075
2: -0.1597649, 0.6353322, -0.0404382, 0.5646217, -0.7243866, 0.6757704
3: -0.2289202, 0.7572249, -0.1053302, 0.5949185, -0.8238387, 0.8625551
4: -0.2726064, 0.8697696, -0.1228297, 0.7481163, -1.0207226, 0.9925992

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24

Time for candidate selection: 2.18 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5694776
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5771162
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0054767, 0.5886192, 0.0739317, 0.5186416, -0.5131649, 0.5146875
1: -0.0523044, 0.7926955, 0.0327510, 0.7059036, -0.7582080, 0.7599446
2: -0.1597649, 0.6353322, -0.0554504, 0.5733088, -0.7330738, 0.6907827
3: -0.2289202, 0.7572249, -0.1144004, 0.6320976, -0.8610178, 0.8716253
4: -0.2726064, 0.8697696, -0.1361425, 0.7930427, -1.0656490, 1.0059121

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24

Time for candidate selection: 2.12 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4835004, upper bound: 0.5694776
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4835004, upper bound: 0.5771162
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0289738, 0.5272163, 0.0677700, 0.5231507, -0.4941769, 0.4594463
1: -0.0271752, 0.6625315, 0.0225216, 0.7127589, -0.7399341, 0.6400099
2: -0.0859941, 0.6136217, -0.0648993, 0.5771931, -0.6631873, 0.6785210
3: -0.1722872, 0.6395842, -0.1257259, 0.6387789, -0.8110661, 0.7653100
4: -0.1796645, 0.7690048, -0.1438800, 0.7990351, -0.9786996, 0.9128848

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5291502, upper bound: 0.5688017
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5182647, upper bound: 0.5689627
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 36

Time for candidate selection: 2.94 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5552753, upper bound: 0.5689627
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5407232, upper bound: 0.5689627
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0166713, 0.5612808, 0.0677700, 0.5231507, -0.5064794, 0.4935108
1: -0.0441760, 0.7052452, 0.0225216, 0.7127589, -0.7569349, 0.6827236
2: -0.1037405, 0.6534984, -0.0648993, 0.5771931, -0.6809336, 0.7183977
3: -0.2023969, 0.6899778, -0.1257259, 0.6387789, -0.8411758, 0.8157037
4: -0.2070733, 0.8241372, -0.1438800, 0.7990351, -1.0061084, 0.9680172

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5177530, upper bound: 0.5688017
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5069878, upper bound: 0.5689627
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24

Time for candidate selection: 2.96 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5407232, upper bound: 0.5689627
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5407232, upper bound: 0.5689627
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0047441, 0.5822396, 0.0982528, 0.3913813, -0.3866372, 0.4839869
1: -0.0574056, 0.7926347, 0.0591810, 0.4799852, -0.5373908, 0.7334538
2: -0.1635008, 0.6239193, 0.0138102, 0.4914325, -0.6549333, 0.6101092
3: -0.2339801, 0.7492464, -0.0678842, 0.4402567, -0.6742368, 0.8171306
4: -0.2717005, 0.8646826, -0.0681658, 0.5925003, -0.8642008, 0.9328483

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24

Time for candidate selection: 1.78 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4879940, upper bound: 0.5735935
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4889665, upper bound: 0.5803225
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0047441, 0.5822396, 0.0946811, 0.4104452, -0.4057012, 0.4875585
1: -0.0574056, 0.7926347, 0.0546263, 0.5232066, -0.5806122, 0.7380084
2: -0.1635008, 0.6239193, -0.0004125, 0.4968947, -0.6603955, 0.6243318
3: -0.2339801, 0.7492464, -0.0763162, 0.4736446, -0.7076247, 0.8255625
4: -0.2717005, 0.8646826, -0.0817058, 0.6355872, -0.9072877, 0.9463884

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24

Time for candidate selection: 1.77 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4900516, upper bound: 0.5553868
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4931412, upper bound: 0.5659867
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0097460, 0.6148499, 0.0982528, 0.3913813, -0.4011272, 0.5165972
1: -0.0750871, 0.8550988, 0.0591810, 0.4799852, -0.5550723, 0.7959179
2: -0.1931111, 0.6471984, 0.0138102, 0.4914325, -0.6845436, 0.6333883
3: -0.2629132, 0.8158054, -0.0678842, 0.4402567, -0.7031699, 0.8836896
4: -0.3095037, 0.9171438, -0.0681658, 0.5925003, -0.9020039, 0.9853096

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24

Time for candidate selection: 1.76 seconds

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5060046, upper bound: 0.5792839
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4915194, upper bound: 0.5679788
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4894430, upper bound: 0.5592957
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0097460, 0.6148499, 0.0946811, 0.4104452, -0.4201912, 0.5201688
1: -0.0750871, 0.8550988, 0.0546263, 0.5232066, -0.5982937, 0.8004725
2: -0.1931111, 0.6471984, -0.0004125, 0.4968947, -0.6900058, 0.6476109
3: -0.2629132, 0.8158054, -0.0763162, 0.4736446, -0.7365577, 0.8921216
4: -0.3095037, 0.9171438, -0.0817058, 0.6355872, -0.9450909, 0.9988496

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24

Time for candidate selection: 1.79 seconds

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5204455, upper bound: 0.5696791
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4921843, upper bound: 0.5497403
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4922520, upper bound: 0.5410681
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0047441, 0.5822396, 0.0902724, 0.4258722, -0.4211282, 0.4919672
1: -0.0574056, 0.7926347, 0.0457680, 0.5438132, -0.6012188, 0.7468667
2: -0.1635008, 0.6239193, -0.0147445, 0.5097890, -0.6732898, 0.6386638
3: -0.2339801, 0.7492464, -0.0928669, 0.4926873, -0.7266675, 0.8421133
4: -0.2717005, 0.8646826, -0.0969031, 0.6412014, -0.9129019, 0.9615856

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24

Time for candidate selection: 1.76 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4865261, upper bound: 0.5752231
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4865261, upper bound: 0.5808136
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0047441, 0.5822396, 0.0876799, 0.4407454, -0.4360013, 0.4945598
1: -0.0574056, 0.7926347, 0.0383623, 0.5784386, -0.6358442, 0.7542725
2: -0.1635008, 0.6239193, -0.0291750, 0.5142469, -0.6777477, 0.6530944
3: -0.2339801, 0.7492464, -0.1023036, 0.5199928, -0.7539729, 0.8515500
4: -0.2717005, 0.8646826, -0.1093452, 0.6781182, -0.9498187, 0.9740278

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24

Time for candidate selection: 1.82 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5008988, upper bound: 0.5752231
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5008988, upper bound: 0.5808136
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0097460, 0.6148499, 0.0902724, 0.4258722, -0.4356182, 0.5245775
1: -0.0750871, 0.8550988, 0.0457680, 0.5438132, -0.6189002, 0.8093308
2: -0.1931111, 0.6471984, -0.0147445, 0.5097890, -0.7029001, 0.6619430
3: -0.2629132, 0.8158054, -0.0928669, 0.4926873, -0.7556005, 0.9086723
4: -0.3095037, 0.9171438, -0.0969031, 0.6412014, -0.9507051, 1.0140469

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24

Time for candidate selection: 1.82 seconds

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4999460, upper bound: 0.5598431
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4999460, upper bound: 0.5702969
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0097460, 0.6148499, 0.0876799, 0.4407454, -0.4504913, 0.5271701
1: -0.0750871, 0.8550988, 0.0383623, 0.5784386, -0.6535257, 0.8167366
2: -0.1931111, 0.6471984, -0.0291750, 0.5142469, -0.7073579, 0.6763735
3: -0.2629132, 0.8158054, -0.1023036, 0.5199928, -0.7829060, 0.9181091
4: -0.3095037, 0.9171438, -0.1093452, 0.6781182, -0.9876219, 1.0264890

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24

Time for candidate selection: 1.80 seconds

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5143186, upper bound: 0.5598431
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5143186, upper bound: 0.5702969
time: 0.47 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0294979, 0.5303939, 0.0899210, 0.4263767, -0.3968788, 0.4404729
1: -0.0272393, 0.6770903, 0.0451424, 0.5446197, -0.5718590, 0.6319479
2: -0.0920497, 0.6105885, -0.0152836, 0.5101860, -0.6022357, 0.6258721
3: -0.1739870, 0.6456757, -0.0934629, 0.4935104, -0.6674973, 0.7391387
4: -0.1849493, 0.7805903, -0.0974107, 0.6418624, -0.8268117, 0.8780010

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 36

Time for candidate selection: 1.76 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5028285, upper bound: 0.5876117
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5132954, upper bound: 0.5624667
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0294979, 0.5303939, 0.0873146, 0.4412901, -0.4117923, 0.4430793
1: -0.0272393, 0.6770903, 0.0379146, 0.5793203, -0.6065596, 0.6391757
2: -0.0920497, 0.6105885, -0.0297543, 0.5146258, -0.6066755, 0.6403428
3: -0.1739870, 0.6456757, -0.1029397, 0.5208486, -0.6948356, 0.7486154
4: -0.1849493, 0.7805903, -0.1098990, 0.6787894, -0.8637387, 0.8904893

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 36

Time for candidate selection: 1.80 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5174631, upper bound: 0.5807620
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5279302, upper bound: 0.5540809
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0056896, 0.5840424, 0.0899210, 0.4263767, -0.4206871, 0.4941214
1: -0.0572673, 0.7667230, 0.0451424, 0.5446197, -0.6018870, 0.7215806
2: -0.1407367, 0.6509148, -0.0152836, 0.5101860, -0.6509227, 0.6661984
3: -0.2236565, 0.7371542, -0.0934629, 0.4935104, -0.7171669, 0.8306171
4: -0.2477118, 0.8606789, -0.0974107, 0.6418624, -0.8895742, 0.9580896

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 36

Time for candidate selection: 1.88 seconds

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5244338, upper bound: 0.5789676
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5269670, upper bound: 0.5605554
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0056896, 0.5840424, 0.0873146, 0.4412901, -0.4356005, 0.4967278
1: -0.0572673, 0.7667230, 0.0379146, 0.5793203, -0.6365876, 0.7288084
2: -0.1407367, 0.6509148, -0.0297543, 0.5146258, -0.6553625, 0.6806691
3: -0.2236565, 0.7371542, -0.1029397, 0.5208486, -0.7445052, 0.8400939
4: -0.2477118, 0.8606789, -0.1098990, 0.6787894, -0.9265012, 0.9705780

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 36

Time for candidate selection: 1.94 seconds

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5390679, upper bound: 0.5697719
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5416019, upper bound: 0.5522098
time: 0.44 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 5.40 seconds
IS_A2_B1_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 5.40
Output dim: 0, lower bound: -0.4683341, upper bound: 0.5694776
IS_A2_B1_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 5.40
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5694776
IS_A2_B1_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 5.40
Output dim: 0, lower bound: -0.4683341, upper bound: 0.5694776
IS_A2_B1_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 5.40
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5694776
IS_A2_B1_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 5.40
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5694776
IS_A2_B1_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 5.40
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5771162
IS_A2_B1_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 5.40
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5694776
IS_A2_B1_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 5.40
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5771162
IS_A2_B1_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 5.40
Output dim: 0, lower bound: -0.4683341, upper bound: 0.5694776
IS_A2_B1_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 5.40
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5694776
IS_A2_B1_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 5.40
Output dim: 0, lower bound: -0.4854941, upper bound: 0.5694776
IS_A2_B1_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 5.40
Output dim: 0, lower bound: -0.4835004, upper bound: 0.5694776
IS_A2_B1_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 5.40
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5694776
IS_A2_B1_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 5.40
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5771162
IS_A2_B1_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 5.40
Output dim: 0, lower bound: -0.4835004, upper bound: 0.5694776
IS_A2_B1_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 5.40
Output dim: 0, lower bound: -0.4835004, upper bound: 0.5771162
IS_A2_B1_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 5.40
Output dim: 0, lower bound: -0.5552753, upper bound: 0.5689627
IS_A2_B1_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 5.40
Output dim: 0, lower bound: -0.5407232, upper bound: 0.5689627
IS_A2_B1_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 5.40
Output dim: 0, lower bound: -0.5407232, upper bound: 0.5689627
IS_A2_B1_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 5.40
Output dim: 0, lower bound: -0.5407232, upper bound: 0.5689627
IS_A2_B1_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 5.40
Output dim: 0, lower bound: -0.4879940, upper bound: 0.5735935
IS_A2_B1_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 5.40
Output dim: 0, lower bound: -0.4889665, upper bound: 0.5803225
IS_A2_B1_A2_B2_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 9, time: 5.40
Output dim: 0, lower bound: -0.4900516, upper bound: 0.5553868
IS_A2_B1_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 5.40
Output dim: 0, lower bound: -0.4931412, upper bound: 0.5659867
IS_A2_B1_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 5.40
Output dim: 0, lower bound: -0.4915194, upper bound: 0.5679788
IS_A2_B1_A2_B2_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 5.40
Output dim: 0, lower bound: -0.4894430, upper bound: 0.5592957
IS_A2_B1_A2_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 5.40
Output dim: 0, lower bound: -0.4921843, upper bound: 0.5497403
IS_A2_B1_A2_B2_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 5.40
Output dim: 0, lower bound: -0.4922520, upper bound: 0.5410681
IS_A2_B1_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 5.40
Output dim: 0, lower bound: -0.4865261, upper bound: 0.5752231
IS_A2_B1_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 5.40
Output dim: 0, lower bound: -0.4865261, upper bound: 0.5808136
IS_A2_B1_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 5.40
Output dim: 0, lower bound: -0.5008988, upper bound: 0.5752231
IS_A2_B1_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 5.40
Output dim: 0, lower bound: -0.5008988, upper bound: 0.5808136
IS_A2_B1_A2_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 5.40
Output dim: 0, lower bound: -0.4999460, upper bound: 0.5598431
IS_A2_B1_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 5.40
Output dim: 0, lower bound: -0.4999460, upper bound: 0.5702969
IS_A2_B1_A2_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 5.40
Output dim: 0, lower bound: -0.5143186, upper bound: 0.5598431
IS_A2_B1_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 5.40
Output dim: 0, lower bound: -0.5143186, upper bound: 0.5702969
IS_A2_B1_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 5.40
Output dim: 0, lower bound: -0.5028285, upper bound: 0.5876117
IS_A2_B1_A2_B2_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 9, time: 5.40
Output dim: 0, lower bound: -0.5132954, upper bound: 0.5624667
IS_A2_B1_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 5.40
Output dim: 0, lower bound: -0.5174631, upper bound: 0.5807620
IS_A2_B1_A2_B2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 9, time: 5.40
Output dim: 0, lower bound: -0.5279302, upper bound: 0.5540809
IS_A2_B1_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 5.40
Output dim: 0, lower bound: -0.5244338, upper bound: 0.5789676
IS_A2_B1_A2_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 5.40
Output dim: 0, lower bound: -0.5269670, upper bound: 0.5605554
IS_A2_B1_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 5.40
Output dim: 0, lower bound: -0.5390679, upper bound: 0.5697719
IS_A2_B1_A2_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 5.40
Output dim: 0, lower bound: -0.5416019, upper bound: 0.5522098

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0186715, 0.5356178, 0.0774577, 0.4984086, -0.4797370, 0.4581601
1: -0.0405200, 0.7052470, 0.0378881, 0.6620049, -0.7025249, 0.6673590
2: -0.1335938, 0.5874144, -0.0404382, 0.5646217, -0.6982155, 0.6278526
3: -0.2105730, 0.6657733, -0.1053302, 0.5949185, -0.8054915, 0.7711036
4: -0.2271247, 0.7838589, -0.1228297, 0.7481163, -0.9752410, 0.9066886

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.60 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5739569
time: 0.46 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5694776
time: 0.47 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0057104, 0.5858070, 0.0774577, 0.4984086, -0.4926982, 0.5083493
1: -0.0520784, 0.7879821, 0.0378881, 0.6620049, -0.7140833, 0.7500941
2: -0.1593977, 0.6317638, -0.0404382, 0.5646217, -0.7240194, 0.6722020
3: -0.2285477, 0.7561934, -0.1053302, 0.5949185, -0.8234662, 0.8615236
4: -0.2714003, 0.8690577, -0.1228297, 0.7481163, -1.0195167, 0.9918873

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.61 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5739569
time: 0.45 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5694776
time: 0.47 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0186715, 0.5356178, 0.0739940, 0.5183154, -0.4996439, 0.4616237
1: -0.0405200, 0.7052470, 0.0328778, 0.7053200, -0.7458400, 0.6723692
2: -0.1335938, 0.5874144, -0.0552860, 0.5731409, -0.7067347, 0.6427004
3: -0.2105730, 0.6657733, -0.1142519, 0.6315680, -0.8421409, 0.7800252
4: -0.2271247, 0.7838589, -0.1360635, 0.7927270, -1.0198517, 0.9199224

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.61 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5694776
time: 0.47 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5694776
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0057104, 0.5858070, 0.0739940, 0.5183154, -0.5126051, 0.5118129
1: -0.0520784, 0.7879821, 0.0328778, 0.7053200, -0.7573984, 0.7551043
2: -0.1593977, 0.6317638, -0.0552860, 0.5731409, -0.7325386, 0.6870499
3: -0.2285477, 0.7561934, -0.1142519, 0.6315680, -0.8601156, 0.8704453
4: -0.2714003, 0.8690577, -0.1360635, 0.7927270, -1.0641273, 1.0051211

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.57 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5694776
time: 0.47 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5694776
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0216482, 0.5324659, 0.0774577, 0.4984086, -0.4767604, 0.4550082
1: -0.0375686, 0.6998085, 0.0378881, 0.6620049, -0.6995735, 0.6619205
2: -0.1289785, 0.5851294, -0.0404382, 0.5646217, -0.6936002, 0.6255676
3: -0.2061760, 0.6599507, -0.1053302, 0.5949185, -0.8010946, 0.7652810
4: -0.2213628, 0.7794645, -0.1228297, 0.7481163, -0.9694791, 0.9022942

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.62 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5739569
time: 0.45 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5694776
time: 0.48 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0054767, 0.5886192, 0.0774577, 0.4984086, -0.4929318, 0.5111616
1: -0.0523044, 0.7926955, 0.0378881, 0.6620049, -0.7143093, 0.7548075
2: -0.1597649, 0.6353322, -0.0404382, 0.5646217, -0.7243866, 0.6757704
3: -0.2289202, 0.7572249, -0.1053302, 0.5949185, -0.8238387, 0.8625551
4: -0.2726064, 0.8697696, -0.1228297, 0.7481163, -1.0207226, 0.9925992

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.62 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5772907
time: 0.44 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5771162
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0216482, 0.5324659, 0.0739940, 0.5183154, -0.4966673, 0.4584718
1: -0.0375686, 0.6998085, 0.0328778, 0.7053200, -0.7428886, 0.6669307
2: -0.1289785, 0.5851294, -0.0552860, 0.5731409, -0.7021194, 0.6404155
3: -0.2061760, 0.6599507, -0.1142519, 0.6315680, -0.8377440, 0.7742026
4: -0.2213628, 0.7794645, -0.1360635, 0.7927270, -1.0140898, 0.9155280

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.57 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5694776
time: 0.47 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5694776
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0054767, 0.5886192, 0.0739940, 0.5183154, -0.5128387, 0.5146252
1: -0.0523044, 0.7926955, 0.0328778, 0.7053200, -0.7576244, 0.7598177
2: -0.1597649, 0.6353322, -0.0552860, 0.5731409, -0.7329058, 0.6906183
3: -0.2289202, 0.7572249, -0.1142519, 0.6315680, -0.8604882, 0.8714768
4: -0.2726064, 0.8697696, -0.1360635, 0.7927270, -1.0653334, 1.0058330

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.60 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5771162
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5771162
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0186715, 0.5356178, 0.0774577, 0.4984086, -0.4797370, 0.4581601
1: -0.0405200, 0.7052470, 0.0378881, 0.6620049, -0.7025249, 0.6673590
2: -0.1335938, 0.5874144, -0.0404382, 0.5646217, -0.6982155, 0.6278526
3: -0.2105730, 0.6657733, -0.1053302, 0.5949185, -0.8054915, 0.7711036
4: -0.2271247, 0.7838589, -0.1228297, 0.7481163, -0.9752410, 0.9066886

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.58 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5739569
time: 0.46 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5694776
time: 0.48 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0057104, 0.5858070, 0.0774577, 0.4984086, -0.4926982, 0.5083493
1: -0.0520784, 0.7879821, 0.0378881, 0.6620049, -0.7140833, 0.7500941
2: -0.1593977, 0.6317638, -0.0404382, 0.5646217, -0.7240194, 0.6722020
3: -0.2285477, 0.7561934, -0.1053302, 0.5949185, -0.8234662, 0.8615236
4: -0.2714003, 0.8690577, -0.1228297, 0.7481163, -1.0195167, 0.9918873

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.62 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5739569
time: 0.46 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5694776
time: 0.48 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0186715, 0.5356178, 0.0739317, 0.5186416, -0.4999701, 0.4616861
1: -0.0405200, 0.7052470, 0.0327510, 0.7059036, -0.7464236, 0.6724961
2: -0.1335938, 0.5874144, -0.0554504, 0.5733088, -0.7069026, 0.6428648
3: -0.2105730, 0.6657733, -0.1144004, 0.6320976, -0.8426706, 0.7801738
4: -0.2271247, 0.7838589, -0.1361425, 0.7930427, -1.0201674, 0.9200014

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.62 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5694776
time: 0.48 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5694776
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0057104, 0.5858070, 0.0739317, 0.5186416, -0.5129312, 0.5118753
1: -0.0520784, 0.7879821, 0.0327510, 0.7059036, -0.7579820, 0.7552311
2: -0.1593977, 0.6317638, -0.0554504, 0.5733088, -0.7327065, 0.6872143
3: -0.2285477, 0.7561934, -0.1144004, 0.6320976, -0.8606453, 0.8705938
4: -0.2714003, 0.8690577, -0.1361425, 0.7930427, -1.0644430, 1.0052001

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.65 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5694776
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5694776
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0216482, 0.5324659, 0.0774577, 0.4984086, -0.4767604, 0.4550082
1: -0.0375686, 0.6998085, 0.0378881, 0.6620049, -0.6995735, 0.6619205
2: -0.1289785, 0.5851294, -0.0404382, 0.5646217, -0.6936002, 0.6255676
3: -0.2061760, 0.6599507, -0.1053302, 0.5949185, -0.8010946, 0.7652810
4: -0.2213628, 0.7794645, -0.1228297, 0.7481163, -0.9694791, 0.9022942

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.66 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5739569
time: 0.45 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5694776
time: 0.47 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0054767, 0.5886192, 0.0774577, 0.4984086, -0.4929318, 0.5111616
1: -0.0523044, 0.7926955, 0.0378881, 0.6620049, -0.7143093, 0.7548075
2: -0.1597649, 0.6353322, -0.0404382, 0.5646217, -0.7243866, 0.6757704
3: -0.2289202, 0.7572249, -0.1053302, 0.5949185, -0.8238387, 0.8625551
4: -0.2726064, 0.8697696, -0.1228297, 0.7481163, -1.0207226, 0.9925992

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.64 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5772907
time: 0.45 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5771162
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0216482, 0.5324659, 0.0739317, 0.5186416, -0.4969934, 0.4585342
1: -0.0375686, 0.6998085, 0.0327510, 0.7059036, -0.7434722, 0.6670576
2: -0.1289785, 0.5851294, -0.0554504, 0.5733088, -0.7022873, 0.6405799
3: -0.2061760, 0.6599507, -0.1144004, 0.6320976, -0.8382736, 0.7743512
4: -0.2213628, 0.7794645, -0.1361425, 0.7930427, -1.0144055, 0.9156070

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.66 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5694776
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5694776
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0054767, 0.5886192, 0.0739317, 0.5186416, -0.5131649, 0.5146875
1: -0.0523044, 0.7926955, 0.0327510, 0.7059036, -0.7582080, 0.7599446
2: -0.1597649, 0.6353322, -0.0554504, 0.5733088, -0.7330738, 0.6907827
3: -0.2289202, 0.7572249, -0.1144004, 0.6320976, -0.8610178, 0.8716253
4: -0.2726064, 0.8697696, -0.1361425, 0.7930427, -1.0656490, 1.0059121

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.68 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4664415, upper bound: 0.5771162
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4835004, upper bound: 0.5771162
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0289738, 0.5272163, 0.0677700, 0.5231507, -0.4941769, 0.4594463
1: -0.0271752, 0.6625315, 0.0225216, 0.7127589, -0.7399341, 0.6400099
2: -0.0859941, 0.6136217, -0.0648993, 0.5771931, -0.6631873, 0.6785210
3: -0.1722872, 0.6395842, -0.1257259, 0.6387789, -0.8110661, 0.7653100
4: -0.1796645, 0.7690048, -0.1438800, 0.7990351, -0.9786996, 0.9128848

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5530363, upper bound: 0.5597668
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 3.20 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5552753, upper bound: 0.5261013
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5552753, upper bound: 0.5689627
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0168582, 0.5608715, 0.0677700, 0.5231507, -0.5062925, 0.4931015
1: -0.0438017, 0.7046244, 0.0225216, 0.7127589, -0.7565606, 0.6821028
2: -0.1034904, 0.6531452, -0.0648993, 0.5771931, -0.6806835, 0.7180445
3: -0.2018650, 0.6891456, -0.1257259, 0.6387789, -0.8406439, 0.8148715
4: -0.2066956, 0.8236248, -0.1438800, 0.7990351, -1.0057306, 0.9675049

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.76 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5407232, upper bound: 0.5261013
time: 0.44 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5407232, upper bound: 0.5689627
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0289738, 0.5272163, 0.0677700, 0.5231507, -0.4941769, 0.4594463
1: -0.0271752, 0.6625315, 0.0225216, 0.7127589, -0.7399341, 0.6400099
2: -0.0859941, 0.6136217, -0.0648993, 0.5771931, -0.6631873, 0.6785210
3: -0.1722872, 0.6395842, -0.1257259, 0.6387789, -0.8110661, 0.7653100
4: -0.1796645, 0.7690048, -0.1438800, 0.7990351, -0.9786996, 0.9128848

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5530363, upper bound: 0.5597668
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 3.15 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5552753, upper bound: 0.5261013
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5552753, upper bound: 0.5689627
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0166713, 0.5612808, 0.0677700, 0.5231507, -0.5064794, 0.4935108
1: -0.0441760, 0.7052452, 0.0225216, 0.7127589, -0.7569349, 0.6827236
2: -0.1037405, 0.6534984, -0.0648993, 0.5771931, -0.6809336, 0.7183977
3: -0.2023969, 0.6899778, -0.1257259, 0.6387789, -0.8411758, 0.8157037
4: -0.2070733, 0.8241372, -0.1438800, 0.7990351, -1.0061084, 0.9680172

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.75 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5407232, upper bound: 0.5261013
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5407232, upper bound: 0.5689627
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0273730, 0.5259886, 0.0982528, 0.3913813, -0.3640083, 0.4277358
1: -0.0302570, 0.6818144, 0.0591810, 0.4799852, -0.5102422, 0.6226335
2: -0.1094030, 0.5890867, 0.0138102, 0.4914325, -0.6008356, 0.5752766
3: -0.1895508, 0.6536713, -0.0678842, 0.4402567, -0.6298075, 0.7215555
4: -0.2105881, 0.7663926, -0.0681658, 0.5925003, -0.8030884, 0.8345584

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 5

Time for candidate selection: 2.29 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4879940, upper bound: 0.5735935
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4879940, upper bound: 0.5735935
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0191859, 0.5786310, 0.0982528, 0.3913813, -0.3721954, 0.4803783
1: -0.0447317, 0.7722255, 0.0591810, 0.4799852, -0.5247170, 0.7130445
2: -0.1387023, 0.6325843, 0.0138102, 0.4914325, -0.6301348, 0.6187741
3: -0.2140281, 0.7337266, -0.0678842, 0.4402567, -0.6542848, 0.8016108
4: -0.2439761, 0.8575616, -0.0681658, 0.5925003, -0.8364764, 0.9257274

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 5

Time for candidate selection: 2.29 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4889663, upper bound: 0.5803225
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4889663, upper bound: 0.5803225
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0191859, 0.5786310, 0.0946811, 0.4104452, -0.3912593, 0.4839499
1: -0.0447317, 0.7722255, 0.0546263, 0.5232066, -0.5679383, 0.7175992
2: -0.1387023, 0.6325843, -0.0004125, 0.4968947, -0.6355970, 0.6329967
3: -0.2140281, 0.7337266, -0.0763162, 0.4736446, -0.6876727, 0.8100427
4: -0.2439761, 0.8575616, -0.0817058, 0.6355872, -0.8795633, 0.9392674

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.31 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4889665, upper bound: 0.5659867
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4931412, upper bound: 0.5659867
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0304394, 0.5232950, 0.0982528, 0.3913813, -0.3609419, 0.4250423
1: -0.0271168, 0.6751002, 0.0591810, 0.4799852, -0.5071020, 0.6159192
2: -0.1008115, 0.5868485, 0.0138102, 0.4914325, -0.5922440, 0.5730383
3: -0.1845073, 0.6496069, -0.0678842, 0.4402567, -0.6247640, 0.7174911
4: -0.2008715, 0.7509959, -0.0681658, 0.5925003, -0.7933717, 0.8191617

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 5

Time for candidate selection: 2.26 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4894430, upper bound: 0.5592957
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4894430, upper bound: 0.5592957
time: 0.44 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0335352, 0.5118269, 0.0902724, 0.4258722, -0.3923370, 0.4215544
1: -0.0245118, 0.6699614, 0.0457680, 0.5438132, -0.5683250, 0.6241934
2: -0.1080940, 0.5659045, -0.0147445, 0.5097890, -0.6178830, 0.5806490
3: -0.1859725, 0.6271144, -0.0928669, 0.4926873, -0.6786598, 0.7199813
4: -0.1974041, 0.7506582, -0.0969031, 0.6412014, -0.8386056, 0.8475613

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.27 seconds

### Candidate
type: B, layer: 3, pos: 41

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4865264, upper bound: 0.5819316
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4865263, upper bound: 0.5819316
time: 0.46 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4865261, upper bound: 0.5752231
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0199920, 0.5662975, 0.0902724, 0.4258722, -0.4058802, 0.4760251
1: -0.0366464, 0.7625916, 0.0457680, 0.5438132, -0.5804596, 0.7168236
2: -0.1364000, 0.6116371, -0.0147445, 0.5097890, -0.6461890, 0.6263816
3: -0.2038792, 0.7202239, -0.0928669, 0.4926873, -0.6965665, 0.8130908
4: -0.2425102, 0.8385897, -0.0969031, 0.6412014, -0.8837116, 0.9354928

Time for backsubstitution: 1.78 seconds
Binary search (step 0): status=Status.UNKNOWN, low=0.0170836, high=0.0994509, mid=0.0994509, abs_max=0.6789970397949219
rel_dist={0: [-0.5950600062778149, 0.5950600062778155]}

## Binary search (step 1) starts
Candidate diff: 0.0582672


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

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
- Time for IS candidates: 0.83 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.83
Output dim: 0, lower bound: -0.5854215, upper bound: 0.5584218
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.83
Output dim: 0, lower bound: -0.5837545, upper bound: 0.5837545

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.0674889, 0.4535027, -0.0339178, 0.6323221, -0.5648332, 0.4874205
1: 0.0201817, 0.5976362, -0.0957146, 0.8296235, -0.8094419, 0.6933507
2: -0.0460193, 0.5242411, -0.1887267, 0.7016845, -0.7477039, 0.7129678
3: -0.1192396, 0.5392005, -0.2802914, 0.8066562, -0.9258958, 0.8194920
4: -0.1227688, 0.6923177, -0.3088930, 0.9303064, -1.0530752, 1.0012107

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5584218, upper bound: 0.5584218
time: 0.35 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5584218, upper bound: 0.5584218
time: 0.35 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0106674, 0.5881721, -0.0365533, 0.6424438, -0.6531112, 0.6247254
1: -0.0680232, 0.7645921, -0.0992057, 0.8423603, -0.9103835, 0.8637978
2: -0.1492940, 0.6635130, -0.1934414, 0.7124153, -0.8617094, 0.8569544
3: -0.2353030, 0.7381678, -0.2859350, 0.8194001, -1.0547031, 1.0241028
4: -0.2566378, 0.8659467, -0.3152393, 0.9457331, -1.2023709, 1.1811860

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

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
- Time for IS candidates: 2.36 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 2.36
Output dim: 0, lower bound: -0.5584218, upper bound: 0.5584218
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 2.36
Output dim: 0, lower bound: -0.5584218, upper bound: 0.5584218
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.36
Output dim: 0, lower bound: -0.5584218, upper bound: 0.5837545
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.36
Output dim: 0, lower bound: -0.5584218, upper bound: 0.5837545

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0106674, 0.5881721, 0.0674889, 0.4535027, -0.4641702, 0.5206832
1: -0.0680232, 0.7645921, 0.0201817, 0.5976362, -0.6656594, 0.7444105
2: -0.1492940, 0.6635130, -0.0460193, 0.5242411, -0.6735351, 0.7095323
3: -0.2353030, 0.7381678, -0.1192396, 0.5392005, -0.7745036, 0.8574073
4: -0.2566378, 0.8659467, -0.1227688, 0.6923177, -0.9489555, 0.9887154

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5450628, upper bound: 0.5640387
time: 0.34 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5566877, upper bound: 0.5820733
time: 0.39 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0106674, 0.5881721, -0.0106674, 0.5881721, -0.5988396, 0.5988396
1: -0.0680232, 0.7645921, -0.0680232, 0.7645921, -0.8326153, 0.8326153
2: -0.1492940, 0.6635130, -0.1492940, 0.6635130, -0.8128070, 0.8128070
3: -0.2353030, 0.7381678, -0.2353030, 0.7381678, -0.9734708, 0.9734708
4: -0.2566378, 0.8659467, -0.2566378, 0.8659467, -1.1225845, 1.1225845

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.11 seconds

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
time: 0.37 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.44 seconds
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 2.44
Output dim: 0, lower bound: -0.5450628, upper bound: 0.5640387
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.44
Output dim: 0, lower bound: -0.5566877, upper bound: 0.5820733
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 2.44
Output dim: 0, lower bound: -0.5450628, upper bound: 0.5640388
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.44
Output dim: 0, lower bound: -0.5566877, upper bound: 0.5820733

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0083361, 0.5719668, 0.0674889, 0.4535027, -0.4451666, 0.5044779
1: -0.0506135, 0.7433118, 0.0201817, 0.5976362, -0.6482497, 0.7231302
2: -0.1297052, 0.6455543, -0.0460193, 0.5242411, -0.6539463, 0.6915736
3: -0.2118729, 0.7126579, -0.1192396, 0.5392005, -0.7510734, 0.8318975
4: -0.2322460, 0.8428500, -0.1227688, 0.6923177, -0.9245638, 0.9656187

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5559137, upper bound: 0.5796032
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5559137, upper bound: 0.5796032
time: 0.38 seconds

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
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5561863, upper bound: 0.5620327
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5561863, upper bound: 0.5620327
time: 0.39 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.44 seconds
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.44
Output dim: 0, lower bound: -0.5559137, upper bound: 0.5796032
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.44
Output dim: 0, lower bound: -0.5559137, upper bound: 0.5796032
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 2.44
Output dim: 0, lower bound: -0.5561863, upper bound: 0.5620327
IS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 2.44
Output dim: 0, lower bound: -0.5561863, upper bound: 0.5620327

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0083361, 0.5719668, 0.0667267, 0.5276489, -0.5193129, 0.5052401
1: -0.0506135, 0.7433118, 0.0209925, 0.7202063, -0.7708198, 0.7223193
2: -0.1297052, 0.6455543, -0.0676523, 0.5800259, -0.7097311, 0.7132066
3: -0.2118729, 0.7126579, -0.1276886, 0.6454976, -0.8573705, 0.8403466
4: -0.2322460, 0.8428500, -0.1461909, 0.8046894, -1.0369354, 0.9890409

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4772043, upper bound: 0.5494218
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4762511, upper bound: 0.5750819
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 36

Time for candidate selection: 3.05 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5539969, upper bound: 0.5729151
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5451910, upper bound: 0.5733778
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0083361, 0.5719668, 0.0789177, 0.4454548, -0.4371188, 0.4930490
1: -0.0506135, 0.7433118, 0.0311321, 0.5851125, -0.6357260, 0.7121797
2: -0.1297052, 0.6455543, -0.0367235, 0.5172547, -0.6469599, 0.6822778
3: -0.2118729, 0.7126579, -0.1111584, 0.5267107, -0.7385836, 0.8238163
4: -0.2322460, 0.8428500, -0.1149590, 0.6824017, -0.9146477, 0.9578090

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4772044, upper bound: 0.5494218
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4762513, upper bound: 0.5750863
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 36

Time for candidate selection: 3.00 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5539970, upper bound: 0.5729151
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5451910, upper bound: 0.5733778
time: 0.40 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 5.37 seconds
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.37
Output dim: 0, lower bound: -0.5539969, upper bound: 0.5729151
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.37
Output dim: 0, lower bound: -0.5451910, upper bound: 0.5733778
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.37
Output dim: 0, lower bound: -0.5539970, upper bound: 0.5729151
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.37
Output dim: 0, lower bound: -0.5451910, upper bound: 0.5733778

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0274241, 0.5315738, 0.0667267, 0.5276489, -0.5002248, 0.4648471
1: -0.0291177, 0.6686592, 0.0209925, 0.7202063, -0.7493240, 0.6476666
2: -0.0887789, 0.6183543, -0.0676523, 0.5800259, -0.6688048, 0.6860067
3: -0.1754975, 0.6463004, -0.1276886, 0.6454976, -0.8209951, 0.7739891
4: -0.1838146, 0.7758477, -0.1461909, 0.8046894, -0.9885041, 0.9220386

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5485251, upper bound: 0.5536649
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5078514, upper bound: 0.5460383
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 3.07 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5498974, upper bound: 0.5266833
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5494941, upper bound: 0.5684957
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0149566, 0.5658405, 0.0667267, 0.5276489, -0.5126923, 0.4991139
1: -0.0463459, 0.7114239, 0.0209925, 0.7202063, -0.7665523, 0.6904314
2: -0.1064991, 0.6583920, -0.0676523, 0.5800259, -0.6865250, 0.7260443
3: -0.2061497, 0.6969915, -0.1276886, 0.6454976, -0.8516473, 0.8246801
4: -0.2113429, 0.8311484, -0.1461909, 0.8046894, -1.0160323, 0.9773393

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5047877, upper bound: 0.5612382
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.75 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5429268, upper bound: 0.5270489
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5408557, upper bound: 0.5688613
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0274241, 0.5315738, 0.0789177, 0.4454548, -0.4180307, 0.4526560
1: -0.0291177, 0.6686592, 0.0311321, 0.5851125, -0.6142302, 0.6375270
2: -0.0887789, 0.6183543, -0.0367235, 0.5172547, -0.6060336, 0.6550778
3: -0.1754975, 0.6463004, -0.1111584, 0.5267107, -0.7022082, 0.7574588
4: -0.1838146, 0.7758477, -0.1149590, 0.6824017, -0.8662163, 0.8908067

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5531423, upper bound: 0.5547464
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5067000, upper bound: 0.5408153
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 3.18 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5508860, upper bound: 0.5251600
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5515525, upper bound: 0.5573571
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0149566, 0.5658405, 0.0789177, 0.4454548, -0.4304982, 0.4869228
1: -0.0463459, 0.7114239, 0.0311321, 0.5851125, -0.6314585, 0.6802918
2: -0.1064991, 0.6583920, -0.0367235, 0.5172547, -0.6237538, 0.6951154
3: -0.2061497, 0.6969915, -0.1111584, 0.5267107, -0.7328604, 0.8081499
4: -0.2113429, 0.8311484, -0.1149590, 0.6824017, -0.8937446, 0.9461074

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5036363, upper bound: 0.5560151
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.83 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5422883, upper bound: 0.5255256
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5408805, upper bound: 0.5573571
time: 0.40 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 5.19 seconds
IS_A2_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.19
Output dim: 0, lower bound: -0.5498974, upper bound: 0.5266833
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.19
Output dim: 0, lower bound: -0.5494941, upper bound: 0.5684957
IS_A2_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.19
Output dim: 0, lower bound: -0.5429268, upper bound: 0.5270489
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.19
Output dim: 0, lower bound: -0.5408557, upper bound: 0.5688613
IS_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.19
Output dim: 0, lower bound: -0.5508860, upper bound: 0.5251600
IS_A2_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.19
Output dim: 0, lower bound: -0.5515525, upper bound: 0.5573571
IS_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.19
Output dim: 0, lower bound: -0.5422883, upper bound: 0.5255256
IS_A2_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 5.19
Output dim: 0, lower bound: -0.5408805, upper bound: 0.5573571

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0274241, 0.5315738, 0.0677700, 0.5231507, -0.4957266, 0.4638038
1: -0.0291177, 0.6686592, 0.0225216, 0.7127589, -0.7418766, 0.6461375
2: -0.0887789, 0.6183543, -0.0648993, 0.5771931, -0.6659720, 0.6832536
3: -0.1754975, 0.6463004, -0.1257259, 0.6387789, -0.8142764, 0.7720263
4: -0.1838146, 0.7758477, -0.1438800, 0.7990351, -0.9828497, 0.9197277

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5231499, upper bound: 0.5683344
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5122826, upper bound: 0.5684957
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 36

Time for candidate selection: 3.13 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5494941, upper bound: 0.5684957
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5408557, upper bound: 0.5684957
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0149566, 0.5658405, 0.0677700, 0.5231507, -0.5081941, 0.4980705
1: -0.0463459, 0.7114239, 0.0225216, 0.7127589, -0.7591048, 0.6889023
2: -0.1064991, 0.6583920, -0.0648993, 0.5771931, -0.6836922, 0.7232913
3: -0.2061497, 0.6969915, -0.1257259, 0.6387789, -0.8449286, 0.8227174
4: -0.2113429, 0.8311484, -0.1438800, 0.7990351, -1.0103780, 0.9750284

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5175789, upper bound: 0.5687000
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5068542, upper bound: 0.5688613
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24

Time for candidate selection: 3.11 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5408557, upper bound: 0.5684957
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5408557, upper bound: 0.5688613
time: 0.39 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 5.50 seconds
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.50
Output dim: 0, lower bound: -0.5494941, upper bound: 0.5684957
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.50
Output dim: 0, lower bound: -0.5408557, upper bound: 0.5684957
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.50
Output dim: 0, lower bound: -0.5408557, upper bound: 0.5684957
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.50
Output dim: 0, lower bound: -0.5408557, upper bound: 0.5688613

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0274241, 0.5315738, 0.0677700, 0.5231507, -0.4957266, 0.4638038
1: -0.0291177, 0.6686592, 0.0225216, 0.7127589, -0.7418766, 0.6461375
2: -0.0887789, 0.6183543, -0.0648993, 0.5771931, -0.6659720, 0.6832536
3: -0.1754975, 0.6463004, -0.1257259, 0.6387789, -0.8142764, 0.7720263
4: -0.1838146, 0.7758477, -0.1438800, 0.7990351, -0.9828497, 0.9197277

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5441967, upper bound: 0.5506275
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.83 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494941, upper bound: 0.5266833
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5494941, upper bound: 0.5684957
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0151531, 0.5654141, 0.0677700, 0.5231507, -0.5079977, 0.4976441
1: -0.0459636, 0.7107825, 0.0225216, 0.7127589, -0.7587225, 0.6882609
2: -0.1062381, 0.6580169, -0.0648993, 0.5771931, -0.6834313, 0.7229162
3: -0.2055742, 0.6961249, -0.1257259, 0.6387789, -0.8443531, 0.8218507
4: -0.2109468, 0.8306049, -0.1438800, 0.7990351, -1.0099819, 0.9744849

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.57 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5288842, upper bound: 0.5572828
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0274241, 0.5315738, 0.0677700, 0.5231507, -0.4957266, 0.4638038
1: -0.0291177, 0.6686592, 0.0225216, 0.7127589, -0.7418766, 0.6461375
2: -0.0887789, 0.6183543, -0.0648993, 0.5771931, -0.6659720, 0.6832536
3: -0.1754975, 0.6463004, -0.1257259, 0.6387789, -0.8142764, 0.7720263
4: -0.1838146, 0.7758477, -0.1438800, 0.7990351, -0.9828497, 0.9197277

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5441967, upper bound: 0.5506275
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.92 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494941, upper bound: 0.5266833
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5494941, upper bound: 0.5684957
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0149566, 0.5658405, 0.0677700, 0.5231507, -0.5081941, 0.4980705
1: -0.0463459, 0.7114239, 0.0225216, 0.7127589, -0.7591048, 0.6889023
2: -0.1064991, 0.6583920, -0.0648993, 0.5771931, -0.6836922, 0.7232913
3: -0.2061497, 0.6969915, -0.1257259, 0.6387789, -0.8449286, 0.8227174
4: -0.2113429, 0.8311484, -0.1438800, 0.7990351, -1.0103780, 0.9750284

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.55 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5408557, upper bound: 0.5270489
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5408557, upper bound: 0.5688613
time: 0.38 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 4.98 seconds
IS_A2_B1_A2_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 4.98
Output dim: 0, lower bound: -0.5494941, upper bound: 0.5266833
IS_A2_B1_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.98
Output dim: 0, lower bound: -0.5494941, upper bound: 0.5684957
IS_A2_B1_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.98
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
IS_A2_B1_A2_B1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 4.98
Output dim: 0, lower bound: -0.5288842, upper bound: 0.5572828
IS_A2_B1_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 4.98
Output dim: 0, lower bound: -0.5494941, upper bound: 0.5266833
IS_A2_B1_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.98
Output dim: 0, lower bound: -0.5494941, upper bound: 0.5684957
IS_A2_B1_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.98
Output dim: 0, lower bound: -0.5408557, upper bound: 0.5270489
IS_A2_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.98
Output dim: 0, lower bound: -0.5408557, upper bound: 0.5688613

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0274241, 0.5315738, 0.0677700, 0.5231507, -0.4957266, 0.4638038
1: -0.0291177, 0.6686592, 0.0225216, 0.7127589, -0.7418766, 0.6461375
2: -0.0887789, 0.6183543, -0.0648993, 0.5771931, -0.6659720, 0.6832536
3: -0.1754975, 0.6463004, -0.1257259, 0.6387789, -0.8142764, 0.7720263
4: -0.1838146, 0.7758477, -0.1438800, 0.7990351, -0.9828497, 0.9197277

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5231499, upper bound: 0.5683344
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5122826, upper bound: 0.5684957
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 36

Time for candidate selection: 3.17 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5494941, upper bound: 0.5684957
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5408557, upper bound: 0.5684957
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0151531, 0.5654141, 0.0778847, 0.4952991, -0.4801461, 0.4875294
1: -0.0459636, 0.7107825, 0.0384705, 0.6570238, -0.7029875, 0.6723120
2: -0.1062381, 0.6580169, -0.0388243, 0.5624917, -0.6687298, 0.6968412
3: -0.2055742, 0.6961249, -0.1042569, 0.5902759, -0.7958500, 0.8003817
4: -0.2109468, 0.8306049, -0.1213620, 0.7441012, -0.9550480, 0.9519669

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4822288, upper bound: 0.5675378
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4786884, upper bound: 0.5676789
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24

Time for candidate selection: 3.25 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0274241, 0.5315738, 0.0677700, 0.5231507, -0.4957266, 0.4638038
1: -0.0291177, 0.6686592, 0.0225216, 0.7127589, -0.7418766, 0.6461375
2: -0.0887789, 0.6183543, -0.0648993, 0.5771931, -0.6659720, 0.6832536
3: -0.1754975, 0.6463004, -0.1257259, 0.6387789, -0.8142764, 0.7720263
4: -0.1838146, 0.7758477, -0.1438800, 0.7990351, -0.9828497, 0.9197277

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5231499, upper bound: 0.5683344
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5122826, upper bound: 0.5684957
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 36

Time for candidate selection: 3.25 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5494941, upper bound: 0.5684957
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5408557, upper bound: 0.5684957
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0149566, 0.5658405, 0.0677700, 0.5231507, -0.5081941, 0.4980705
1: -0.0463459, 0.7114239, 0.0225216, 0.7127589, -0.7591048, 0.6889023
2: -0.1064991, 0.6583920, -0.0648993, 0.5771931, -0.6836922, 0.7232913
3: -0.2061497, 0.6969915, -0.1257259, 0.6387789, -0.8449286, 0.8227174
4: -0.2113429, 0.8311484, -0.1438800, 0.7990351, -1.0103780, 0.9750284

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5175789, upper bound: 0.5687000
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5068542, upper bound: 0.5688613
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24

Time for candidate selection: 3.26 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5408557, upper bound: 0.5684957
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5408557, upper bound: 0.5688613
time: 0.39 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 5.72 seconds
IS_A2_B1_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 5.72
Output dim: 0, lower bound: -0.5494941, upper bound: 0.5684957
IS_A2_B1_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 5.72
Output dim: 0, lower bound: -0.5408557, upper bound: 0.5684957
IS_A2_B1_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 5.72
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
IS_A2_B1_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 5.72
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
IS_A2_B1_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 5.72
Output dim: 0, lower bound: -0.5494941, upper bound: 0.5684957
IS_A2_B1_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 5.72
Output dim: 0, lower bound: -0.5408557, upper bound: 0.5684957
IS_A2_B1_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 5.72
Output dim: 0, lower bound: -0.5408557, upper bound: 0.5684957
IS_A2_B1_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 5.72
Output dim: 0, lower bound: -0.5408557, upper bound: 0.5688613

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0274241, 0.5315738, 0.0677700, 0.5231507, -0.4957266, 0.4638038
1: -0.0291177, 0.6686592, 0.0225216, 0.7127589, -0.7418766, 0.6461375
2: -0.0887789, 0.6183543, -0.0648993, 0.5771931, -0.6659720, 0.6832536
3: -0.1754975, 0.6463004, -0.1257259, 0.6387789, -0.8142764, 0.7720263
4: -0.1838146, 0.7758477, -0.1438800, 0.7990351, -0.9828497, 0.9197277

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5441967, upper bound: 0.5506275
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.94 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494941, upper bound: 0.5266833
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5494941, upper bound: 0.5684957
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0151531, 0.5654141, 0.0677700, 0.5231507, -0.5079977, 0.4976441
1: -0.0459636, 0.7107825, 0.0225216, 0.7127589, -0.7587225, 0.6882609
2: -0.1062381, 0.6580169, -0.0648993, 0.5771931, -0.6834313, 0.7229162
3: -0.2055742, 0.6961249, -0.1257259, 0.6387789, -0.8443531, 0.8218507
4: -0.2109468, 0.8306049, -0.1438800, 0.7990351, -1.0099819, 0.9744849

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.67 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5288842, upper bound: 0.5572828
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0274241, 0.5315738, 0.0778847, 0.4952991, -0.4678750, 0.4536891
1: -0.0291177, 0.6686592, 0.0384705, 0.6570238, -0.6861416, 0.6301886
2: -0.0887789, 0.6183543, -0.0388243, 0.5624917, -0.6512706, 0.6571786
3: -0.1754975, 0.6463004, -0.1042569, 0.5902759, -0.7657734, 0.7505573
4: -0.1838146, 0.7758477, -0.1213620, 0.7441012, -0.9279158, 0.8972096

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5193840, upper bound: 0.5498383
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.95 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5237767, upper bound: 0.5250030
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5279497, upper bound: 0.5676799
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0151531, 0.5654141, 0.0778847, 0.4952991, -0.4801461, 0.4875294
1: -0.0459636, 0.7107825, 0.0384705, 0.6570238, -0.7029875, 0.6723120
2: -0.1062381, 0.6580169, -0.0388243, 0.5624917, -0.6687298, 0.6968412
3: -0.2055742, 0.6961249, -0.1042569, 0.5902759, -0.7958500, 0.8003817
4: -0.2109468, 0.8306049, -0.1213620, 0.7441012, -0.9550480, 0.9519669

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.62 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5572828
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0274241, 0.5315738, 0.0677700, 0.5231507, -0.4957266, 0.4638038
1: -0.0291177, 0.6686592, 0.0225216, 0.7127589, -0.7418766, 0.6461375
2: -0.0887789, 0.6183543, -0.0648993, 0.5771931, -0.6659720, 0.6832536
3: -0.1754975, 0.6463004, -0.1257259, 0.6387789, -0.8142764, 0.7720263
4: -0.1838146, 0.7758477, -0.1438800, 0.7990351, -0.9828497, 0.9197277

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5441967, upper bound: 0.5506275
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.97 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494941, upper bound: 0.5266833
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5494941, upper bound: 0.5684957
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0151531, 0.5654141, 0.0677700, 0.5231507, -0.5079977, 0.4976441
1: -0.0459636, 0.7107825, 0.0225216, 0.7127589, -0.7587225, 0.6882609
2: -0.1062381, 0.6580169, -0.0648993, 0.5771931, -0.6834313, 0.7229162
3: -0.2055742, 0.6961249, -0.1257259, 0.6387789, -0.8443531, 0.8218507
4: -0.2109468, 0.8306049, -0.1438800, 0.7990351, -1.0099819, 0.9744849

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.63 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5288842, upper bound: 0.5572828
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0274241, 0.5315738, 0.0677700, 0.5231507, -0.4957266, 0.4638038
1: -0.0291177, 0.6686592, 0.0225216, 0.7127589, -0.7418766, 0.6461375
2: -0.0887789, 0.6183543, -0.0648993, 0.5771931, -0.6659720, 0.6832536
3: -0.1754975, 0.6463004, -0.1257259, 0.6387789, -0.8142764, 0.7720263
4: -0.1838146, 0.7758477, -0.1438800, 0.7990351, -0.9828497, 0.9197277

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5441967, upper bound: 0.5506275
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.96 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494941, upper bound: 0.5266833
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5494941, upper bound: 0.5684957
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0149566, 0.5658405, 0.0677700, 0.5231507, -0.5081941, 0.4980705
1: -0.0463459, 0.7114239, 0.0225216, 0.7127589, -0.7591048, 0.6889023
2: -0.1064991, 0.6583920, -0.0648993, 0.5771931, -0.6836922, 0.7232913
3: -0.2061497, 0.6969915, -0.1257259, 0.6387789, -0.8449286, 0.8227174
4: -0.2113429, 0.8311484, -0.1438800, 0.7990351, -1.0103780, 0.9750284

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.66 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5408557, upper bound: 0.5270489
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5408557, upper bound: 0.5688613
time: 0.41 seconds

## Summary of splitting at layer (split count: 9)
- Time for IS candidates: 5.13 seconds
IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 10, time: 5.13
Output dim: 0, lower bound: -0.5494941, upper bound: 0.5266833
IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 5.13
Output dim: 0, lower bound: -0.5494941, upper bound: 0.5684957
IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 5.13
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 10, time: 5.13
Output dim: 0, lower bound: -0.5288842, upper bound: 0.5572828
IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 10, time: 5.13
Output dim: 0, lower bound: -0.5237767, upper bound: 0.5250030
IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 5.13
Output dim: 0, lower bound: -0.5279497, upper bound: 0.5676799
IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 5.13
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 10, time: 5.13
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5572828
IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 10, time: 5.13
Output dim: 0, lower bound: -0.5494941, upper bound: 0.5266833
IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 5.13
Output dim: 0, lower bound: -0.5494941, upper bound: 0.5684957
IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 5.13
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 10, time: 5.13
Output dim: 0, lower bound: -0.5288842, upper bound: 0.5572828
IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 10, time: 5.13
Output dim: 0, lower bound: -0.5494941, upper bound: 0.5266833
IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 5.13
Output dim: 0, lower bound: -0.5494941, upper bound: 0.5684957
IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 5.13
Output dim: 0, lower bound: -0.5408557, upper bound: 0.5270489
IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 5.13
Output dim: 0, lower bound: -0.5408557, upper bound: 0.5688613

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0274241, 0.5315738, 0.0677700, 0.5231507, -0.4957266, 0.4638038
1: -0.0291177, 0.6686592, 0.0225216, 0.7127589, -0.7418766, 0.6461375
2: -0.0887789, 0.6183543, -0.0648993, 0.5771931, -0.6659720, 0.6832536
3: -0.1754975, 0.6463004, -0.1257259, 0.6387789, -0.8142764, 0.7720263
4: -0.1838146, 0.7758477, -0.1438800, 0.7990351, -0.9828497, 0.9197277

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5231499, upper bound: 0.5683344
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5122826, upper bound: 0.5684957
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 36

Time for candidate selection: 3.27 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5494941, upper bound: 0.5684957
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5408557, upper bound: 0.5684957
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0151531, 0.5654141, 0.0778847, 0.4952991, -0.4801461, 0.4875294
1: -0.0459636, 0.7107825, 0.0384705, 0.6570238, -0.7029875, 0.6723120
2: -0.1062381, 0.6580169, -0.0388243, 0.5624917, -0.6687298, 0.6968412
3: -0.2055742, 0.6961249, -0.1042569, 0.5902759, -0.7958500, 0.8003817
4: -0.2109468, 0.8306049, -0.1213620, 0.7441012, -0.9550480, 0.9519669

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4822288, upper bound: 0.5675378
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4786884, upper bound: 0.5676790
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24

Time for candidate selection: 3.32 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0274241, 0.5315738, 0.0778847, 0.4952991, -0.4678750, 0.4536891
1: -0.0291177, 0.6686592, 0.0384705, 0.6570238, -0.6861416, 0.6301886
2: -0.0887789, 0.6183543, -0.0388243, 0.5624917, -0.6512706, 0.6571786
3: -0.1754975, 0.6463004, -0.1042569, 0.5902759, -0.7657734, 0.7505573
4: -0.1838146, 0.7758477, -0.1213620, 0.7441012, -0.9279158, 0.8972096

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4900980, upper bound: 0.5675378
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4865026, upper bound: 0.5676789
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 36

Time for candidate selection: 3.36 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5279497, upper bound: 0.5676799
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
time: 0.45 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0151531, 0.5654141, 0.0778847, 0.4952991, -0.4801461, 0.4875294
1: -0.0459636, 0.7107825, 0.0384705, 0.6570238, -0.7029875, 0.6723120
2: -0.1062381, 0.6580169, -0.0388243, 0.5624917, -0.6687298, 0.6968412
3: -0.2055742, 0.6961249, -0.1042569, 0.5902759, -0.7958500, 0.8003817
4: -0.2109468, 0.8306049, -0.1213620, 0.7441012, -0.9550480, 0.9519669

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4822288, upper bound: 0.5675378
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4786884, upper bound: 0.5676790
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24

Time for candidate selection: 3.40 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0274241, 0.5315738, 0.0677700, 0.5231507, -0.4957266, 0.4638038
1: -0.0291177, 0.6686592, 0.0225216, 0.7127589, -0.7418766, 0.6461375
2: -0.0887789, 0.6183543, -0.0648993, 0.5771931, -0.6659720, 0.6832536
3: -0.1754975, 0.6463004, -0.1257259, 0.6387789, -0.8142764, 0.7720263
4: -0.1838146, 0.7758477, -0.1438800, 0.7990351, -0.9828497, 0.9197277

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5231499, upper bound: 0.5683344
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5122826, upper bound: 0.5684957
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 36

Time for candidate selection: 3.30 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5494941, upper bound: 0.5684957
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5408557, upper bound: 0.5684957
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0151531, 0.5654141, 0.0778847, 0.4952991, -0.4801461, 0.4875294
1: -0.0459636, 0.7107825, 0.0384705, 0.6570238, -0.7029875, 0.6723120
2: -0.1062381, 0.6580169, -0.0388243, 0.5624917, -0.6687298, 0.6968412
3: -0.2055742, 0.6961249, -0.1042569, 0.5902759, -0.7958500, 0.8003817
4: -0.2109468, 0.8306049, -0.1213620, 0.7441012, -0.9550480, 0.9519669

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4822288, upper bound: 0.5675378
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4786884, upper bound: 0.5676790
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24

Time for candidate selection: 3.33 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0274241, 0.5315738, 0.0677700, 0.5231507, -0.4957266, 0.4638038
1: -0.0291177, 0.6686592, 0.0225216, 0.7127589, -0.7418766, 0.6461375
2: -0.0887789, 0.6183543, -0.0648993, 0.5771931, -0.6659720, 0.6832536
3: -0.1754975, 0.6463004, -0.1257259, 0.6387789, -0.8142764, 0.7720263
4: -0.1838146, 0.7758477, -0.1438800, 0.7990351, -0.9828497, 0.9197277

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5231499, upper bound: 0.5683344
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5122826, upper bound: 0.5684957
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 36

Time for candidate selection: 3.35 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5494941, upper bound: 0.5684957
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5408557, upper bound: 0.5684957
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0149566, 0.5658405, 0.0677700, 0.5231507, -0.5081941, 0.4980705
1: -0.0463459, 0.7114239, 0.0225216, 0.7127589, -0.7591048, 0.6889023
2: -0.1064991, 0.6583920, -0.0648993, 0.5771931, -0.6836922, 0.7232913
3: -0.2061497, 0.6969915, -0.1257259, 0.6387789, -0.8449286, 0.8227174
4: -0.2113429, 0.8311484, -0.1438800, 0.7990351, -1.0103780, 0.9750284

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5175789, upper bound: 0.5687000
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5068542, upper bound: 0.5688613
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24

Time for candidate selection: 3.36 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5408557, upper bound: 0.5684957
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5408557, upper bound: 0.5688613
time: 0.41 seconds

## Summary of splitting at layer (split count: 10)
- Time for IS candidates: 5.88 seconds
IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 11, time: 5.88
Output dim: 0, lower bound: -0.5494941, upper bound: 0.5684957
IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 11, time: 5.88
Output dim: 0, lower bound: -0.5408557, upper bound: 0.5684957
IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 11, time: 5.88
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 5.88
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 11, time: 5.88
Output dim: 0, lower bound: -0.5279497, upper bound: 0.5676799
IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 11, time: 5.88
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 11, time: 5.88
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 5.88
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 11, time: 5.88
Output dim: 0, lower bound: -0.5494941, upper bound: 0.5684957
IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 11, time: 5.88
Output dim: 0, lower bound: -0.5408557, upper bound: 0.5684957
IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 11, time: 5.88
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 5.88
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 11, time: 5.88
Output dim: 0, lower bound: -0.5494941, upper bound: 0.5684957
IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 11, time: 5.88
Output dim: 0, lower bound: -0.5408557, upper bound: 0.5684957
IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 11, time: 5.88
Output dim: 0, lower bound: -0.5408557, upper bound: 0.5684957
IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 5.88
Output dim: 0, lower bound: -0.5408557, upper bound: 0.5688613

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0274241, 0.5315738, 0.0677700, 0.5231507, -0.4957266, 0.4638038
1: -0.0291177, 0.6686592, 0.0225216, 0.7127589, -0.7418766, 0.6461375
2: -0.0887789, 0.6183543, -0.0648993, 0.5771931, -0.6659720, 0.6832536
3: -0.1754975, 0.6463004, -0.1257259, 0.6387789, -0.8142764, 0.7720263
4: -0.1838146, 0.7758477, -0.1438800, 0.7990351, -0.9828497, 0.9197277

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5441967, upper bound: 0.5506275
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 3.05 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494941, upper bound: 0.5266833
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5494941, upper bound: 0.5684957
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0151531, 0.5654141, 0.0677700, 0.5231507, -0.5079977, 0.4976441
1: -0.0459636, 0.7107825, 0.0225216, 0.7127589, -0.7587225, 0.6882609
2: -0.1062381, 0.6580169, -0.0648993, 0.5771931, -0.6834313, 0.7229162
3: -0.2055742, 0.6961249, -0.1257259, 0.6387789, -0.8443531, 0.8218507
4: -0.2109468, 0.8306049, -0.1438800, 0.7990351, -1.0099819, 0.9744849

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.76 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5288842, upper bound: 0.5572828
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0274241, 0.5315738, 0.0778847, 0.4952991, -0.4678750, 0.4536891
1: -0.0291177, 0.6686592, 0.0384705, 0.6570238, -0.6861416, 0.6301886
2: -0.0887789, 0.6183543, -0.0388243, 0.5624917, -0.6512706, 0.6571786
3: -0.1754975, 0.6463004, -0.1042569, 0.5902759, -0.7657734, 0.7505573
4: -0.1838146, 0.7758477, -0.1213620, 0.7441012, -0.9279158, 0.8972096

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5193840, upper bound: 0.5498383
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 3.04 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5237767, upper bound: 0.5250030
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5279497, upper bound: 0.5676799
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0151531, 0.5654141, 0.0778847, 0.4952991, -0.4801461, 0.4875294
1: -0.0459636, 0.7107825, 0.0384705, 0.6570238, -0.7029875, 0.6723120
2: -0.1062381, 0.6580169, -0.0388243, 0.5624917, -0.6687298, 0.6968412
3: -0.2055742, 0.6961249, -0.1042569, 0.5902759, -0.7958500, 0.8003817
4: -0.2109468, 0.8306049, -0.1213620, 0.7441012, -0.9550480, 0.9519669

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.78 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5572828
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0274241, 0.5315738, 0.0778847, 0.4952991, -0.4678750, 0.4536891
1: -0.0291177, 0.6686592, 0.0384705, 0.6570238, -0.6861416, 0.6301886
2: -0.0887789, 0.6183543, -0.0388243, 0.5624917, -0.6512706, 0.6571786
3: -0.1754975, 0.6463004, -0.1042569, 0.5902759, -0.7657734, 0.7505573
4: -0.1838146, 0.7758477, -0.1213620, 0.7441012, -0.9279158, 0.8972096

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5193827, upper bound: 0.5498377
time: 0.49 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 3.18 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5237767, upper bound: 0.5250030
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5279497, upper bound: 0.5676799
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0151531, 0.5654141, 0.0778847, 0.4952991, -0.4801461, 0.4875294
1: -0.0459636, 0.7107825, 0.0384705, 0.6570238, -0.7029875, 0.6723120
2: -0.1062381, 0.6580169, -0.0388243, 0.5624917, -0.6687298, 0.6968412
3: -0.2055742, 0.6961249, -0.1042569, 0.5902759, -0.7958500, 0.8003817
4: -0.2109468, 0.8306049, -0.1213620, 0.7441012, -0.9550480, 0.9519669

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.76 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5158992, upper bound: 0.5250030
time: 0.45 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0274241, 0.5315738, 0.0778847, 0.4952991, -0.4678750, 0.4536891
1: -0.0291177, 0.6686592, 0.0384705, 0.6570238, -0.6861416, 0.6301886
2: -0.0887789, 0.6183543, -0.0388243, 0.5624917, -0.6512706, 0.6571786
3: -0.1754975, 0.6463004, -0.1042569, 0.5902759, -0.7657734, 0.7505573
4: -0.1838146, 0.7758477, -0.1213620, 0.7441012, -0.9279158, 0.8972096

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5193840, upper bound: 0.5498383
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 3.13 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5237767, upper bound: 0.5250030
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5279497, upper bound: 0.5676799
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0151531, 0.5654141, 0.0778847, 0.4952991, -0.4801461, 0.4875294
1: -0.0459636, 0.7107825, 0.0384705, 0.6570238, -0.7029875, 0.6723120
2: -0.1062381, 0.6580169, -0.0388243, 0.5624917, -0.6687298, 0.6968412
3: -0.2055742, 0.6961249, -0.1042569, 0.5902759, -0.7958500, 0.8003817
4: -0.2109468, 0.8306049, -0.1213620, 0.7441012, -0.9550480, 0.9519669

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.77 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5572828
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0274241, 0.5315738, 0.0677700, 0.5231507, -0.4957266, 0.4638038
1: -0.0291177, 0.6686592, 0.0225216, 0.7127589, -0.7418766, 0.6461375
2: -0.0887789, 0.6183543, -0.0648993, 0.5771931, -0.6659720, 0.6832536
3: -0.1754975, 0.6463004, -0.1257259, 0.6387789, -0.8142764, 0.7720263
4: -0.1838146, 0.7758477, -0.1438800, 0.7990351, -0.9828497, 0.9197277

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5441967, upper bound: 0.5506275
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 3.08 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494941, upper bound: 0.5266833
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5494941, upper bound: 0.5684957
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0151531, 0.5654141, 0.0677700, 0.5231507, -0.5079977, 0.4976441
1: -0.0459636, 0.7107825, 0.0225216, 0.7127589, -0.7587225, 0.6882609
2: -0.1062381, 0.6580169, -0.0648993, 0.5771931, -0.6834313, 0.7229162
3: -0.2055742, 0.6961249, -0.1257259, 0.6387789, -0.8443531, 0.8218507
4: -0.2109468, 0.8306049, -0.1438800, 0.7990351, -1.0099819, 0.9744849

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.82 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5288842, upper bound: 0.5572828
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0274241, 0.5315738, 0.0778847, 0.4952991, -0.4678750, 0.4536891
1: -0.0291177, 0.6686592, 0.0384705, 0.6570238, -0.6861416, 0.6301886
2: -0.0887789, 0.6183543, -0.0388243, 0.5624917, -0.6512706, 0.6571786
3: -0.1754975, 0.6463004, -0.1042569, 0.5902759, -0.7657734, 0.7505573
4: -0.1838146, 0.7758477, -0.1213620, 0.7441012, -0.9279158, 0.8972096

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5193827, upper bound: 0.5498377
time: 0.44 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 3.15 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5237767, upper bound: 0.5250030
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5279497, upper bound: 0.5676799
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0151531, 0.5654141, 0.0778847, 0.4952991, -0.4801461, 0.4875294
1: -0.0459636, 0.7107825, 0.0384705, 0.6570238, -0.7029875, 0.6723120
2: -0.1062381, 0.6580169, -0.0388243, 0.5624917, -0.6687298, 0.6968412
3: -0.2055742, 0.6961249, -0.1042569, 0.5902759, -0.7958500, 0.8003817
4: -0.2109468, 0.8306049, -0.1213620, 0.7441012, -0.9550480, 0.9519669

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.84 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5572828
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0274241, 0.5315738, 0.0677700, 0.5231507, -0.4957266, 0.4638038
1: -0.0291177, 0.6686592, 0.0225216, 0.7127589, -0.7418766, 0.6461375
2: -0.0887789, 0.6183543, -0.0648993, 0.5771931, -0.6659720, 0.6832536
3: -0.1754975, 0.6463004, -0.1257259, 0.6387789, -0.8142764, 0.7720263
4: -0.1838146, 0.7758477, -0.1438800, 0.7990351, -0.9828497, 0.9197277

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5441967, upper bound: 0.5506275
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 3.12 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494941, upper bound: 0.5266833
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5494941, upper bound: 0.5684957
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0151531, 0.5654141, 0.0677700, 0.5231507, -0.5079977, 0.4976441
1: -0.0459636, 0.7107825, 0.0225216, 0.7127589, -0.7587225, 0.6882609
2: -0.1062381, 0.6580169, -0.0648993, 0.5771931, -0.6834313, 0.7229162
3: -0.2055742, 0.6961249, -0.1257259, 0.6387789, -0.8443531, 0.8218507
4: -0.2109468, 0.8306049, -0.1438800, 0.7990351, -1.0099819, 0.9744849

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.82 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5288842, upper bound: 0.5572828
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0274241, 0.5315738, 0.0677700, 0.5231507, -0.4957266, 0.4638038
1: -0.0291177, 0.6686592, 0.0225216, 0.7127589, -0.7418766, 0.6461375
2: -0.0887789, 0.6183543, -0.0648993, 0.5771931, -0.6659720, 0.6832536
3: -0.1754975, 0.6463004, -0.1257259, 0.6387789, -0.8142764, 0.7720263
4: -0.1838146, 0.7758477, -0.1438800, 0.7990351, -0.9828497, 0.9197277

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5441967, upper bound: 0.5506275
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 3.14 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494941, upper bound: 0.5266833
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5494941, upper bound: 0.5684957
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0149566, 0.5658405, 0.0677700, 0.5231507, -0.5081941, 0.4980705
1: -0.0463459, 0.7114239, 0.0225216, 0.7127589, -0.7591048, 0.6889023
2: -0.1064991, 0.6583920, -0.0648993, 0.5771931, -0.6836922, 0.7232913
3: -0.2061497, 0.6969915, -0.1257259, 0.6387789, -0.8449286, 0.8227174
4: -0.2113429, 0.8311484, -0.1438800, 0.7990351, -1.0103780, 0.9750284

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.77 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5408557, upper bound: 0.5270489
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5408557, upper bound: 0.5688613
time: 0.42 seconds

## Summary of splitting at layer (split count: 11)
- Time for IS candidates: 5.33 seconds
IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 12, time: 5.33
Output dim: 0, lower bound: -0.5494941, upper bound: 0.5266833
IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 12, time: 5.33
Output dim: 0, lower bound: -0.5494941, upper bound: 0.5684957
IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 12, time: 5.33
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 12, time: 5.33
Output dim: 0, lower bound: -0.5288842, upper bound: 0.5572828
IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 12, time: 5.33
Output dim: 0, lower bound: -0.5237767, upper bound: 0.5250030
IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 12, time: 5.33
Output dim: 0, lower bound: -0.5279497, upper bound: 0.5676799
IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 12, time: 5.33
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 12, time: 5.33
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5572828
IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 12, time: 5.33
Output dim: 0, lower bound: -0.5237767, upper bound: 0.5250030
IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 12, time: 5.33
Output dim: 0, lower bound: -0.5279497, upper bound: 0.5676799
IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 12, time: 5.33
Output dim: 0, lower bound: -0.5158992, upper bound: 0.5250030
IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 5.33
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 12, time: 5.33
Output dim: 0, lower bound: -0.5237767, upper bound: 0.5250030
IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 12, time: 5.33
Output dim: 0, lower bound: -0.5279497, upper bound: 0.5676799
IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 12, time: 5.33
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 12, time: 5.33
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5572828
IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 12, time: 5.33
Output dim: 0, lower bound: -0.5494941, upper bound: 0.5266833
IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 12, time: 5.33
Output dim: 0, lower bound: -0.5494941, upper bound: 0.5684957
IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 12, time: 5.33
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 12, time: 5.33
Output dim: 0, lower bound: -0.5288842, upper bound: 0.5572828
IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 12, time: 5.33
Output dim: 0, lower bound: -0.5237767, upper bound: 0.5250030
IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 12, time: 5.33
Output dim: 0, lower bound: -0.5279497, upper bound: 0.5676799
IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 12, time: 5.33
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 12, time: 5.33
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5572828
IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 12, time: 5.33
Output dim: 0, lower bound: -0.5494941, upper bound: 0.5266833
IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 12, time: 5.33
Output dim: 0, lower bound: -0.5494941, upper bound: 0.5684957
IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 12, time: 5.33
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 12, time: 5.33
Output dim: 0, lower bound: -0.5288842, upper bound: 0.5572828
IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 12, time: 5.33
Output dim: 0, lower bound: -0.5494941, upper bound: 0.5266833
IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 12, time: 5.33
Output dim: 0, lower bound: -0.5494941, upper bound: 0.5684957
IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 12, time: 5.33
Output dim: 0, lower bound: -0.5408557, upper bound: 0.5270489
IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 5.33
Output dim: 0, lower bound: -0.5408557, upper bound: 0.5688613

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0274241, 0.5315738, 0.0677700, 0.5231507, -0.4957266, 0.4638038
1: -0.0291177, 0.6686592, 0.0225216, 0.7127589, -0.7418766, 0.6461375
2: -0.0887789, 0.6183543, -0.0648993, 0.5771931, -0.6659720, 0.6832536
3: -0.1754975, 0.6463004, -0.1257259, 0.6387789, -0.8142764, 0.7720263
4: -0.1838146, 0.7758477, -0.1438800, 0.7990351, -0.9828497, 0.9197277

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5231499, upper bound: 0.5683344
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5122826, upper bound: 0.5684957
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 36

Time for candidate selection: 3.49 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5494941, upper bound: 0.5684957
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5408557, upper bound: 0.5684957
time: 0.45 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0151531, 0.5654141, 0.0778847, 0.4952991, -0.4801461, 0.4875294
1: -0.0459636, 0.7107825, 0.0384705, 0.6570238, -0.7029875, 0.6723120
2: -0.1062381, 0.6580169, -0.0388243, 0.5624917, -0.6687298, 0.6968412
3: -0.2055742, 0.6961249, -0.1042569, 0.5902759, -0.7958500, 0.8003817
4: -0.2109468, 0.8306049, -0.1213620, 0.7441012, -0.9550480, 0.9519669

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4822288, upper bound: 0.5675378
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4786884, upper bound: 0.5676789
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24

Time for candidate selection: 3.48 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0274241, 0.5315738, 0.0778847, 0.4952991, -0.4678750, 0.4536891
1: -0.0291177, 0.6686592, 0.0384705, 0.6570238, -0.6861416, 0.6301886
2: -0.0887789, 0.6183543, -0.0388243, 0.5624917, -0.6512706, 0.6571786
3: -0.1754975, 0.6463004, -0.1042569, 0.5902759, -0.7657734, 0.7505573
4: -0.1838146, 0.7758477, -0.1213620, 0.7441012, -0.9279158, 0.8972096

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4900980, upper bound: 0.5675378
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4865026, upper bound: 0.5676789
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 36

Time for candidate selection: 3.42 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5279497, upper bound: 0.5676799
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0151531, 0.5654141, 0.0778847, 0.4952991, -0.4801461, 0.4875294
1: -0.0459636, 0.7107825, 0.0384705, 0.6570238, -0.7029875, 0.6723120
2: -0.1062381, 0.6580169, -0.0388243, 0.5624917, -0.6687298, 0.6968412
3: -0.2055742, 0.6961249, -0.1042569, 0.5902759, -0.7958500, 0.8003817
4: -0.2109468, 0.8306049, -0.1213620, 0.7441012, -0.9550480, 0.9519669

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4822288, upper bound: 0.5675378
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4786884, upper bound: 0.5676789
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24

Time for candidate selection: 3.42 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0274241, 0.5315738, 0.0778847, 0.4952991, -0.4678750, 0.4536891
1: -0.0291177, 0.6686592, 0.0384705, 0.6570238, -0.6861416, 0.6301886
2: -0.0887789, 0.6183543, -0.0388243, 0.5624917, -0.6512706, 0.6571786
3: -0.1754975, 0.6463004, -0.1042569, 0.5902759, -0.7657734, 0.7505573
4: -0.1838146, 0.7758477, -0.1213620, 0.7441012, -0.9279158, 0.8972096

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4900980, upper bound: 0.5675378
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4865026, upper bound: 0.5676789
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 36

Time for candidate selection: 3.49 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5279497, upper bound: 0.5676799
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0151531, 0.5654141, 0.0778847, 0.4952991, -0.4801461, 0.4875294
1: -0.0459636, 0.7107825, 0.0384705, 0.6570238, -0.7029875, 0.6723120
2: -0.1062381, 0.6580169, -0.0388243, 0.5624917, -0.6687298, 0.6968412
3: -0.2055742, 0.6961249, -0.1042569, 0.5902759, -0.7958500, 0.8003817
4: -0.2109468, 0.8306049, -0.1213620, 0.7441012, -0.9550480, 0.9519669

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4822288, upper bound: 0.5675378
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4786884, upper bound: 0.5676789
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24

Time for candidate selection: 3.49 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0274241, 0.5315738, 0.0778847, 0.4952991, -0.4678750, 0.4536891
1: -0.0291177, 0.6686592, 0.0384705, 0.6570238, -0.6861416, 0.6301886
2: -0.0887789, 0.6183543, -0.0388243, 0.5624917, -0.6512706, 0.6571786
3: -0.1754975, 0.6463004, -0.1042569, 0.5902759, -0.7657734, 0.7505573
4: -0.1838146, 0.7758477, -0.1213620, 0.7441012, -0.9279158, 0.8972096

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4900980, upper bound: 0.5675378
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4865026, upper bound: 0.5676789
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 36

Time for candidate selection: 3.43 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5279497, upper bound: 0.5676799
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0151531, 0.5654141, 0.0778847, 0.4952991, -0.4801461, 0.4875294
1: -0.0459636, 0.7107825, 0.0384705, 0.6570238, -0.7029875, 0.6723120
2: -0.1062381, 0.6580169, -0.0388243, 0.5624917, -0.6687298, 0.6968412
3: -0.2055742, 0.6961249, -0.1042569, 0.5902759, -0.7958500, 0.8003817
4: -0.2109468, 0.8306049, -0.1213620, 0.7441012, -0.9550480, 0.9519669

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4822288, upper bound: 0.5675378
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4786884, upper bound: 0.5676789
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24

Time for candidate selection: 3.44 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0274241, 0.5315738, 0.0677700, 0.5231507, -0.4957266, 0.4638038
1: -0.0291177, 0.6686592, 0.0225216, 0.7127589, -0.7418766, 0.6461375
2: -0.0887789, 0.6183543, -0.0648993, 0.5771931, -0.6659720, 0.6832536
3: -0.1754975, 0.6463004, -0.1257259, 0.6387789, -0.8142764, 0.7720263
4: -0.1838146, 0.7758477, -0.1438800, 0.7990351, -0.9828497, 0.9197277

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5231499, upper bound: 0.5683344
time: 0.44 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5122826, upper bound: 0.5684957
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 36

Time for candidate selection: 3.54 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5494941, upper bound: 0.5684957
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5408557, upper bound: 0.5684957
time: 0.45 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0151531, 0.5654141, 0.0778847, 0.4952991, -0.4801461, 0.4875294
1: -0.0459636, 0.7107825, 0.0384705, 0.6570238, -0.7029875, 0.6723120
2: -0.1062381, 0.6580169, -0.0388243, 0.5624917, -0.6687298, 0.6968412
3: -0.2055742, 0.6961249, -0.1042569, 0.5902759, -0.7958500, 0.8003817
4: -0.2109468, 0.8306049, -0.1213620, 0.7441012, -0.9550480, 0.9519669

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4822288, upper bound: 0.5675378
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4786884, upper bound: 0.5676789
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24

Time for candidate selection: 3.52 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0274241, 0.5315738, 0.0778847, 0.4952991, -0.4678750, 0.4536891
1: -0.0291177, 0.6686592, 0.0384705, 0.6570238, -0.6861416, 0.6301886
2: -0.0887789, 0.6183543, -0.0388243, 0.5624917, -0.6512706, 0.6571786
3: -0.1754975, 0.6463004, -0.1042569, 0.5902759, -0.7657734, 0.7505573
4: -0.1838146, 0.7758477, -0.1213620, 0.7441012, -0.9279158, 0.8972096

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4900980, upper bound: 0.5675378
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4865026, upper bound: 0.5676789
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 36

Time for candidate selection: 3.52 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5279497, upper bound: 0.5676799
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0151531, 0.5654141, 0.0778847, 0.4952991, -0.4801461, 0.4875294
1: -0.0459636, 0.7107825, 0.0384705, 0.6570238, -0.7029875, 0.6723120
2: -0.1062381, 0.6580169, -0.0388243, 0.5624917, -0.6687298, 0.6968412
3: -0.2055742, 0.6961249, -0.1042569, 0.5902759, -0.7958500, 0.8003817
4: -0.2109468, 0.8306049, -0.1213620, 0.7441012, -0.9550480, 0.9519669

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4822288, upper bound: 0.5675378
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4786884, upper bound: 0.5676789
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24

Time for candidate selection: 3.51 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0274241, 0.5315738, 0.0677700, 0.5231507, -0.4957266, 0.4638038
1: -0.0291177, 0.6686592, 0.0225216, 0.7127589, -0.7418766, 0.6461375
2: -0.0887789, 0.6183543, -0.0648993, 0.5771931, -0.6659720, 0.6832536
3: -0.1754975, 0.6463004, -0.1257259, 0.6387789, -0.8142764, 0.7720263
4: -0.1838146, 0.7758477, -0.1438800, 0.7990351, -0.9828497, 0.9197277

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5231499, upper bound: 0.5683344
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5122826, upper bound: 0.5684957
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 36

Time for candidate selection: 3.58 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5494941, upper bound: 0.5684957
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5408557, upper bound: 0.5684957
time: 0.45 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0151531, 0.5654141, 0.0778847, 0.4952991, -0.4801461, 0.4875294
1: -0.0459636, 0.7107825, 0.0384705, 0.6570238, -0.7029875, 0.6723120
2: -0.1062381, 0.6580169, -0.0388243, 0.5624917, -0.6687298, 0.6968412
3: -0.2055742, 0.6961249, -0.1042569, 0.5902759, -0.7958500, 0.8003817
4: -0.2109468, 0.8306049, -0.1213620, 0.7441012, -0.9550480, 0.9519669

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4822288, upper bound: 0.5675378
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4786884, upper bound: 0.5676789
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24

Time for candidate selection: 3.53 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
time: 0.44 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0274241, 0.5315738, 0.0677700, 0.5231507, -0.4957266, 0.4638038
1: -0.0291177, 0.6686592, 0.0225216, 0.7127589, -0.7418766, 0.6461375
2: -0.0887789, 0.6183543, -0.0648993, 0.5771931, -0.6659720, 0.6832536
3: -0.1754975, 0.6463004, -0.1257259, 0.6387789, -0.8142764, 0.7720263
4: -0.1838146, 0.7758477, -0.1438800, 0.7990351, -0.9828497, 0.9197277

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5231499, upper bound: 0.5683344
time: 0.44 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5122826, upper bound: 0.5684957
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 36

Time for candidate selection: 3.56 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5494941, upper bound: 0.5684957
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5408557, upper bound: 0.5684957
time: 0.45 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0149566, 0.5658405, 0.0677700, 0.5231507, -0.5081941, 0.4980705
1: -0.0463459, 0.7114239, 0.0225216, 0.7127589, -0.7591048, 0.6889023
2: -0.1064991, 0.6583920, -0.0648993, 0.5771931, -0.6836922, 0.7232913
3: -0.2061497, 0.6969915, -0.1257259, 0.6387789, -0.8449286, 0.8227174
4: -0.2113429, 0.8311484, -0.1438800, 0.7990351, -1.0103780, 0.9750284

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5175789, upper bound: 0.5687000
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5068542, upper bound: 0.5688613
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24

Time for candidate selection: 3.62 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5408557, upper bound: 0.5684957
time: 0.44 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5408557, upper bound: 0.5688613
time: 0.43 seconds

## Summary of splitting at layer (split count: 12)
- Time for IS candidates: 6.30 seconds
IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 13, time: 6.30
Output dim: 0, lower bound: -0.5494941, upper bound: 0.5684957
IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 13, time: 6.30
Output dim: 0, lower bound: -0.5408557, upper bound: 0.5684957
IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 13, time: 6.30
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 13, time: 6.30
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 13, time: 6.30
Output dim: 0, lower bound: -0.5279497, upper bound: 0.5676799
IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 13, time: 6.30
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 13, time: 6.30
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 13, time: 6.30
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 13, time: 6.30
Output dim: 0, lower bound: -0.5279497, upper bound: 0.5676799
IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 13, time: 6.30
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 13, time: 6.30
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 13, time: 6.30
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 13, time: 6.30
Output dim: 0, lower bound: -0.5279497, upper bound: 0.5676799
IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 13, time: 6.30
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 13, time: 6.30
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 13, time: 6.30
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 13, time: 6.30
Output dim: 0, lower bound: -0.5494941, upper bound: 0.5684957
IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 13, time: 6.30
Output dim: 0, lower bound: -0.5408557, upper bound: 0.5684957
IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 13, time: 6.30
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 13, time: 6.30
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 13, time: 6.30
Output dim: 0, lower bound: -0.5279497, upper bound: 0.5676799
IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 13, time: 6.30
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 13, time: 6.30
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 13, time: 6.30
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 13, time: 6.30
Output dim: 0, lower bound: -0.5494941, upper bound: 0.5684957
IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 13, time: 6.30
Output dim: 0, lower bound: -0.5408557, upper bound: 0.5684957
IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 13, time: 6.30
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 13, time: 6.30
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 13, time: 6.30
Output dim: 0, lower bound: -0.5494941, upper bound: 0.5684957
IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 13, time: 6.30
Output dim: 0, lower bound: -0.5408557, upper bound: 0.5684957
IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 13, time: 6.30
Output dim: 0, lower bound: -0.5408557, upper bound: 0.5684957
IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 13, time: 6.30
Output dim: 0, lower bound: -0.5408557, upper bound: 0.5688613

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0274241, 0.5315738, 0.0677700, 0.5231507, -0.4957266, 0.4638038
1: -0.0291177, 0.6686592, 0.0225216, 0.7127589, -0.7418766, 0.6461375
2: -0.0887789, 0.6183543, -0.0648993, 0.5771931, -0.6659720, 0.6832536
3: -0.1754975, 0.6463004, -0.1257259, 0.6387789, -0.8142764, 0.7720263
4: -0.1838146, 0.7758477, -0.1438800, 0.7990351, -0.9828497, 0.9197277

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5441967, upper bound: 0.5506275
time: 0.47 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 3.34 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494941, upper bound: 0.5266833
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5494941, upper bound: 0.5684957
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0151531, 0.5654141, 0.0677700, 0.5231507, -0.5079977, 0.4976441
1: -0.0459636, 0.7107825, 0.0225216, 0.7127589, -0.7587225, 0.6882609
2: -0.1062381, 0.6580169, -0.0648993, 0.5771931, -0.6834313, 0.7229162
3: -0.2055742, 0.6961249, -0.1257259, 0.6387789, -0.8443531, 0.8218507
4: -0.2109468, 0.8306049, -0.1438800, 0.7990351, -1.0099819, 0.9744849

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.98 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5288842, upper bound: 0.5572828
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0274241, 0.5315738, 0.0778847, 0.4952991, -0.4678750, 0.4536891
1: -0.0291177, 0.6686592, 0.0384705, 0.6570238, -0.6861416, 0.6301886
2: -0.0887789, 0.6183543, -0.0388243, 0.5624917, -0.6512706, 0.6571786
3: -0.1754975, 0.6463004, -0.1042569, 0.5902759, -0.7657734, 0.7505573
4: -0.1838146, 0.7758477, -0.1213620, 0.7441012, -0.9279158, 0.8972096

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5193840, upper bound: 0.5498383
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 3.26 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5237767, upper bound: 0.5250030
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5279497, upper bound: 0.5676799
time: 0.45 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0151531, 0.5654141, 0.0778847, 0.4952991, -0.4801461, 0.4875294
1: -0.0459636, 0.7107825, 0.0384705, 0.6570238, -0.7029875, 0.6723120
2: -0.1062381, 0.6580169, -0.0388243, 0.5624917, -0.6687298, 0.6968412
3: -0.2055742, 0.6961249, -0.1042569, 0.5902759, -0.7958500, 0.8003817
4: -0.2109468, 0.8306049, -0.1213620, 0.7441012, -0.9550480, 0.9519669

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.95 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5572828
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0274241, 0.5315738, 0.0778847, 0.4952991, -0.4678750, 0.4536891
1: -0.0291177, 0.6686592, 0.0384705, 0.6570238, -0.6861416, 0.6301886
2: -0.0887789, 0.6183543, -0.0388243, 0.5624917, -0.6512706, 0.6571786
3: -0.1754975, 0.6463004, -0.1042569, 0.5902759, -0.7657734, 0.7505573
4: -0.1838146, 0.7758477, -0.1213620, 0.7441012, -0.9279158, 0.8972096

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5193827, upper bound: 0.5498377
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 3.28 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5237767, upper bound: 0.5250030
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5279497, upper bound: 0.5676799
time: 0.45 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0151531, 0.5654141, 0.0778847, 0.4952991, -0.4801461, 0.4875294
1: -0.0459636, 0.7107825, 0.0384705, 0.6570238, -0.7029875, 0.6723120
2: -0.1062381, 0.6580169, -0.0388243, 0.5624917, -0.6687298, 0.6968412
3: -0.2055742, 0.6961249, -0.1042569, 0.5902759, -0.7958500, 0.8003817
4: -0.2109468, 0.8306049, -0.1213620, 0.7441012, -0.9550480, 0.9519669

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.95 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5158992, upper bound: 0.5250030
time: 0.46 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5175135, upper bound: 0.5676799
time: 0.44 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0274241, 0.5315738, 0.0778847, 0.4952991, -0.4678750, 0.4536891
1: -0.0291177, 0.6686592, 0.0384705, 0.6570238, -0.6861416, 0.6301886
2: -0.0887789, 0.6183543, -0.0388243, 0.5624917, -0.6512706, 0.6571786
3: -0.1754975, 0.6463004, -0.1042569, 0.5902759, -0.7657734, 0.7505573
4: -0.1838146, 0.7758477, -0.1213620, 0.7441012, -0.9279158, 0.8972096

Time for backsubstitution: 1.75 seconds
Binary search (step 1): status=Status.UNKNOWN, low=0.0170836, high=0.0582672, mid=0.0582672, abs_max=0.6789970397949219
rel_dist={0: [-0.5863374219570315, 0.5863374219570305]}

## Binary search (step 2) starts
Candidate diff: 0.0376754


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
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

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5559249, upper bound: 0.5559249
time: 0.36 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5559249, upper bound: 0.5559249
time: 0.39 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0106674, 0.5881721, -0.0306828, 0.6307013, -0.6413687, 0.6188549
1: -0.0680232, 0.7645921, -0.0919231, 0.8260971, -0.8941203, 0.8565152
2: -0.1492940, 0.6635130, -0.1838701, 0.7013786, -0.8506727, 0.8473831
3: -0.2353030, 0.7381678, -0.2745254, 0.8015700, -1.0368731, 1.0126932
4: -0.2566378, 0.8659467, -0.3016382, 0.9287875, -1.1854253, 1.1675849

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.11 seconds

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
- Time for IS candidates: 2.37 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 2.37
Output dim: 0, lower bound: -0.5559249, upper bound: 0.5559249
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 2.37
Output dim: 0, lower bound: -0.5559249, upper bound: 0.5559249
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.37
Output dim: 0, lower bound: -0.5559249, upper bound: 0.5758081
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.37
Output dim: 0, lower bound: -0.5559249, upper bound: 0.5758081

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0106674, 0.5881721, 0.0674889, 0.4535027, -0.4641702, 0.5206832
1: -0.0680232, 0.7645921, 0.0201817, 0.5976362, -0.6656594, 0.7444105
2: -0.1492940, 0.6635130, -0.0460193, 0.5242411, -0.6735351, 0.7095323
3: -0.2353030, 0.7381678, -0.1192396, 0.5392005, -0.7745036, 0.8574073
4: -0.2566378, 0.8659467, -0.1227688, 0.6923177, -0.9489555, 0.9887154

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5450628, upper bound: 0.5614455
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5541648, upper bound: 0.5739817
time: 0.36 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0106674, 0.5881721, -0.0106674, 0.5881721, -0.5988396, 0.5988396
1: -0.0680232, 0.7645921, -0.0680232, 0.7645921, -0.8326153, 0.8326153
2: -0.1492940, 0.6635130, -0.1492940, 0.6635130, -0.8128070, 0.8128070
3: -0.2353030, 0.7381678, -0.2353030, 0.7381678, -0.9734708, 0.9734708
4: -0.2566378, 0.8659467, -0.2566378, 0.8659467, -1.1225845, 1.1225845

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5450628, upper bound: 0.5614455
time: 0.50 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5541648, upper bound: 0.5739818
time: 0.39 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.62 seconds
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 2.62
Output dim: 0, lower bound: -0.5450628, upper bound: 0.5614455
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.62
Output dim: 0, lower bound: -0.5541648, upper bound: 0.5739817
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 2.62
Output dim: 0, lower bound: -0.5450628, upper bound: 0.5614455
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.62
Output dim: 0, lower bound: -0.5541648, upper bound: 0.5739818

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0083361, 0.5719668, 0.0674889, 0.4535027, -0.4451666, 0.5044779
1: -0.0506135, 0.7433118, 0.0201817, 0.5976362, -0.6482497, 0.7231302
2: -0.1297052, 0.6455543, -0.0460193, 0.5242411, -0.6539463, 0.6915736
3: -0.2118729, 0.7126579, -0.1192396, 0.5392005, -0.7510734, 0.8318975
4: -0.2322460, 0.8428500, -0.1227688, 0.6923177, -0.9245638, 0.9656187

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5533205, upper bound: 0.5683151
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5533205, upper bound: 0.5739817
time: 0.35 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0083361, 0.5719668, -0.0106674, 0.5881721, -0.5798361, 0.5826342
1: -0.0506135, 0.7433118, -0.0680232, 0.7645921, -0.8152056, 0.8113350
2: -0.1297052, 0.6455543, -0.1492940, 0.6635130, -0.7932182, 0.7948483
3: -0.2118729, 0.7126579, -0.2353030, 0.7381678, -0.9500406, 0.9479610
4: -0.2322460, 0.8428500, -0.2566378, 0.8659467, -1.0981927, 1.0994878

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5535089, upper bound: 0.5603068
time: 0.39 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5535089, upper bound: 0.5739817
time: 0.37 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.47 seconds
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -0.5533205, upper bound: 0.5683151
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -0.5533205, upper bound: 0.5739817
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 2.47
Output dim: 0, lower bound: -0.5535089, upper bound: 0.5603068
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -0.5535089, upper bound: 0.5739817

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0083361, 0.5719668, 0.0667267, 0.5276489, -0.5193129, 0.5052401
1: -0.0506135, 0.7433118, 0.0209925, 0.7202063, -0.7708198, 0.7223193
2: -0.1297052, 0.6455543, -0.0676523, 0.5800259, -0.7097311, 0.7132066
3: -0.2118729, 0.7126579, -0.1276886, 0.6454976, -0.8573705, 0.8403466
4: -0.2322460, 0.8428500, -0.1461909, 0.8046894, -1.0369354, 0.9890409

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 36

Time for candidate selection: 2.34 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5509670, upper bound: 0.5629631
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5447842, upper bound: 0.5644678
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0083361, 0.5719668, 0.0789177, 0.4454548, -0.4371188, 0.4930490
1: -0.0506135, 0.7433118, 0.0311321, 0.5851125, -0.6357260, 0.7121797
2: -0.1297052, 0.6455543, -0.0367235, 0.5172547, -0.6469599, 0.6822778
3: -0.2118729, 0.7126579, -0.1111584, 0.5267107, -0.7385836, 0.8238163
4: -0.2322460, 0.8428500, -0.1149590, 0.6824017, -0.9146477, 0.9578090

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 36

Time for candidate selection: 2.39 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5509672, upper bound: 0.5629631
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5447842, upper bound: 0.5645301
time: 0.38 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0083361, 0.5719668, 0.0083361, 0.5719668, -0.5636307, 0.5636307
1: -0.0506135, 0.7433118, -0.0506135, 0.7433118, -0.7939253, 0.7939253
2: -0.1297052, 0.6455543, -0.1297052, 0.6455543, -0.7752595, 0.7752595
3: -0.2118729, 0.7126579, -0.2118729, 0.7126579, -0.9245308, 0.9245308
4: -0.2322460, 0.8428500, -0.2322460, 0.8428500, -1.0750960, 1.0750960

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 36

Time for candidate selection: 2.38 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5509683, upper bound: 0.5611244
time: 0.39 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5472408, upper bound: 0.5623527
time: 0.42 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.76 seconds
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.76
Output dim: 0, lower bound: -0.5509670, upper bound: 0.5629631
IS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.76
Output dim: 0, lower bound: -0.5447842, upper bound: 0.5644678
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.76
Output dim: 0, lower bound: -0.5509672, upper bound: 0.5629631
IS_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.76
Output dim: 0, lower bound: -0.5447842, upper bound: 0.5645301
IS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.76
Output dim: 0, lower bound: -0.5509683, upper bound: 0.5611244
IS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.76
Output dim: 0, lower bound: -0.5472408, upper bound: 0.5623527
Binary search (step 2): status=Status.VERIFIED, low=0.0376754, high=0.0582672, mid=0.0376754, abs_max=0.6789970397949219
rel_dist={0: [-0.5783685500496472, 0.5783685500496476]}

## Binary search (step 3) starts
Candidate diff: 0.0479713


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5807665, upper bound: 0.5572586
time: 0.37 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5805637, upper bound: 0.5805636
time: 0.39 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.92 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.92
Output dim: 0, lower bound: -0.5807665, upper bound: 0.5572586
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.92
Output dim: 0, lower bound: -0.5805637, upper bound: 0.5805636

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.0674889, 0.4535027, -0.0271196, 0.6065611, -0.5390722, 0.4806224
1: 0.0201817, 0.5976362, -0.0876218, 0.7966388, -0.7764572, 0.6852580
2: -0.0460193, 0.5242411, -0.1764928, 0.6746604, -0.7206798, 0.7007339
3: -0.1192396, 0.5392005, -0.2658013, 0.7742159, -0.8934555, 0.8050019
4: -0.1227688, 0.6923177, -0.2924895, 0.8913207, -1.0140895, 0.9848073

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5572586, upper bound: 0.5572586
time: 0.33 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5572586, upper bound: 0.5572586
time: 0.33 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0106674, 0.5881721, -0.0355616, 0.6404293, -0.6510968, 0.6237337
1: -0.0680232, 0.7645921, -0.0979711, 0.8396053, -0.9076285, 0.8625632
2: -0.1492940, 0.6635130, -0.1918182, 0.7104836, -0.8597776, 0.8553312
3: -0.2353030, 0.7381678, -0.2840012, 0.8163543, -1.0516573, 1.0221690
4: -0.2566378, 0.8659467, -0.3129047, 0.9428010, -1.1994388, 1.1788514

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5572586, upper bound: 0.5805637
time: 0.34 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5572586, upper bound: 0.5805637
time: 0.37 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.39 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 2.39
Output dim: 0, lower bound: -0.5572586, upper bound: 0.5572586
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 2.39
Output dim: 0, lower bound: -0.5572586, upper bound: 0.5572586
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.39
Output dim: 0, lower bound: -0.5572586, upper bound: 0.5805637
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.39
Output dim: 0, lower bound: -0.5572586, upper bound: 0.5805637

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0106674, 0.5881721, 0.0674889, 0.4535027, -0.4641702, 0.5206832
1: -0.0680232, 0.7645921, 0.0201817, 0.5976362, -0.6656594, 0.7444105
2: -0.1492940, 0.6635130, -0.0460193, 0.5242411, -0.6735351, 0.7095323
3: -0.2353030, 0.7381678, -0.1192396, 0.5392005, -0.7745036, 0.8574073
4: -0.2566378, 0.8659467, -0.1227688, 0.6923177, -0.9489555, 0.9887154

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5450628, upper bound: 0.5627421
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5554263, upper bound: 0.5789002
time: 0.37 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0106674, 0.5881721, -0.0106674, 0.5881721, -0.5988396, 0.5988396
1: -0.0680232, 0.7645921, -0.0680232, 0.7645921, -0.8326153, 0.8326153
2: -0.1492940, 0.6635130, -0.1492940, 0.6635130, -0.8128070, 0.8128070
3: -0.2353030, 0.7381678, -0.2353030, 0.7381678, -0.9734708, 0.9734708
4: -0.2566378, 0.8659467, -0.2566378, 0.8659467, -1.1225845, 1.1225845

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5450628, upper bound: 0.5627421
time: 0.40 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5554263, upper bound: 0.5789003
time: 0.36 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.44 seconds
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 2.44
Output dim: 0, lower bound: -0.5450628, upper bound: 0.5627421
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.44
Output dim: 0, lower bound: -0.5554263, upper bound: 0.5789002
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 2.44
Output dim: 0, lower bound: -0.5450628, upper bound: 0.5627421
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.44
Output dim: 0, lower bound: -0.5554263, upper bound: 0.5789003

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0083361, 0.5719668, 0.0674889, 0.4535027, -0.4451666, 0.5044779
1: -0.0506135, 0.7433118, 0.0201817, 0.5976362, -0.6482497, 0.7231302
2: -0.1297052, 0.6455543, -0.0460193, 0.5242411, -0.6539463, 0.6915736
3: -0.2118729, 0.7126579, -0.1192396, 0.5392005, -0.7510734, 0.8318975
4: -0.2322460, 0.8428500, -0.1227688, 0.6923177, -0.9245638, 0.9656187

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5546171, upper bound: 0.5740405
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5546171, upper bound: 0.5740405
time: 0.37 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0083361, 0.5719668, -0.0106674, 0.5881721, -0.5798361, 0.5826342
1: -0.0506135, 0.7433118, -0.0680232, 0.7645921, -0.8152056, 0.8113350
2: -0.1297052, 0.6455543, -0.1492940, 0.6635130, -0.7932182, 0.7948483
3: -0.2118729, 0.7126579, -0.2353030, 0.7381678, -0.9500406, 0.9479610
4: -0.2322460, 0.8428500, -0.2566378, 0.8659467, -1.0981927, 1.0994878

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5548539, upper bound: 0.5611953
time: 0.39 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5548539, upper bound: 0.5789003
time: 0.37 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.45 seconds
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.45
Output dim: 0, lower bound: -0.5546171, upper bound: 0.5740405
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.45
Output dim: 0, lower bound: -0.5546171, upper bound: 0.5740405
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 2.45
Output dim: 0, lower bound: -0.5548539, upper bound: 0.5611953
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.45
Output dim: 0, lower bound: -0.5548539, upper bound: 0.5789003

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0083361, 0.5719668, 0.0667267, 0.5276489, -0.5193129, 0.5052401
1: -0.0506135, 0.7433118, 0.0209925, 0.7202063, -0.7708198, 0.7223193
2: -0.1297052, 0.6455543, -0.0676523, 0.5800259, -0.7097311, 0.7132066
3: -0.2118729, 0.7126579, -0.1276886, 0.6454976, -0.8573705, 0.8403466
4: -0.2322460, 0.8428500, -0.1461909, 0.8046894, -1.0369354, 0.9890409

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4757431, upper bound: 0.5132224
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 36

Time for candidate selection: 2.74 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5524967, upper bound: 0.5680936
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5426809, upper bound: 0.5692177
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0083361, 0.5719668, 0.0789177, 0.4454548, -0.4371188, 0.4930490
1: -0.0506135, 0.7433118, 0.0311321, 0.5851125, -0.6357260, 0.7121797
2: -0.1297052, 0.6455543, -0.0367235, 0.5172547, -0.6469599, 0.6822778
3: -0.2118729, 0.7126579, -0.1111584, 0.5267107, -0.7385836, 0.8238163
4: -0.2322460, 0.8428500, -0.1149590, 0.6824017, -0.9146477, 0.9578090

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4757432, upper bound: 0.5132224
time: 0.45 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 36

Time for candidate selection: 2.86 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5524969, upper bound: 0.5680936
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5450775, upper bound: 0.5692177
time: 0.38 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0083361, 0.5719668, 0.0083361, 0.5719668, -0.5636307, 0.5636307
1: -0.0506135, 0.7433118, -0.0506135, 0.7433118, -0.7939253, 0.7939253
2: -0.1297052, 0.6455543, -0.1297052, 0.6455543, -0.7752595, 0.7752595
3: -0.2118729, 0.7126579, -0.2118729, 0.7126579, -0.9245308, 0.9245308
4: -0.2322460, 0.8428500, -0.2322460, 0.8428500, -1.0750960, 1.0750960

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 36

Time for candidate selection: 2.38 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5525248, upper bound: 0.5630892
time: 0.44 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5477904, upper bound: 0.5647645
time: 0.40 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.84 seconds
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.84
Output dim: 0, lower bound: -0.5524967, upper bound: 0.5680936
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.84
Output dim: 0, lower bound: -0.5426809, upper bound: 0.5692177
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.84
Output dim: 0, lower bound: -0.5524969, upper bound: 0.5680936
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.84
Output dim: 0, lower bound: -0.5450775, upper bound: 0.5692177
IS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.84
Output dim: 0, lower bound: -0.5525248, upper bound: 0.5630892
IS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.84
Output dim: 0, lower bound: -0.5477904, upper bound: 0.5647645

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0274241, 0.5315738, 0.0667267, 0.5276489, -0.5002248, 0.4648142
1: -0.0291177, 0.6686592, 0.0209925, 0.7202063, -0.7493240, 0.6476666
2: -0.0887789, 0.6183543, -0.0676523, 0.5800259, -0.6688048, 0.6860067
3: -0.1754975, 0.6463004, -0.1276886, 0.6454976, -0.8209951, 0.7739891
4: -0.1838146, 0.7758477, -0.1461909, 0.8046894, -0.9885041, 0.9220386

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5428692, upper bound: 0.5472210
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5078514, upper bound: 0.5415595
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 3.00 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5483821, upper bound: 0.5257600
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5479939, upper bound: 0.5641729
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0149566, 0.5658405, 0.0667267, 0.5276489, -0.5126923, 0.4990118
1: -0.0463459, 0.7114239, 0.0209925, 0.7202063, -0.7665523, 0.6904314
2: -0.1064991, 0.6583920, -0.0676523, 0.5800259, -0.6865250, 0.7260443
3: -0.2061497, 0.6969915, -0.1276886, 0.6454976, -0.8516473, 0.8246801
4: -0.2113429, 0.8311484, -0.1461909, 0.8046894, -1.0160323, 0.9773393

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5047877, upper bound: 0.5567594
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.84 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5406326, upper bound: 0.5268704
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5407788, upper bound: 0.5646289
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0274241, 0.5315738, 0.0789177, 0.4454548, -0.4180307, 0.4526560
1: -0.0291177, 0.6686592, 0.0311321, 0.5851125, -0.6142302, 0.6375270
2: -0.0887789, 0.6183543, -0.0367235, 0.5172547, -0.6060336, 0.6550778
3: -0.1754975, 0.6463004, -0.1111584, 0.5267107, -0.7022082, 0.7574588
4: -0.1838146, 0.7758477, -0.1149590, 0.6824017, -0.8662163, 0.8908067

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5511595, upper bound: 0.5486188
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5067000, upper bound: 0.5386487
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 3.19 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5494213, upper bound: 0.5242450
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5500877, upper bound: 0.5569198
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0149566, 0.5658405, 0.0789177, 0.4454548, -0.4304982, 0.4869228
1: -0.0463459, 0.7114239, 0.0311321, 0.5851125, -0.6314585, 0.6802918
2: -0.1064991, 0.6583920, -0.0367235, 0.5172547, -0.6237538, 0.6951154
3: -0.2061497, 0.6969915, -0.1111584, 0.5267107, -0.7328604, 0.8081499
4: -0.2113429, 0.8311484, -0.1149590, 0.6824017, -0.8937446, 0.9461074

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5036363, upper bound: 0.5538486
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.82 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5420289, upper bound: 0.5251611
time: 0.48 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5383014, upper bound: 0.5569198
time: 0.41 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 5.36 seconds
IS_A2_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.36
Output dim: 0, lower bound: -0.5483821, upper bound: 0.5257600
IS_A2_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.36
Output dim: 0, lower bound: -0.5479939, upper bound: 0.5641729
IS_A2_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.36
Output dim: 0, lower bound: -0.5406326, upper bound: 0.5268704
IS_A2_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 5.36
Output dim: 0, lower bound: -0.5407788, upper bound: 0.5646289
IS_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.36
Output dim: 0, lower bound: -0.5494213, upper bound: 0.5242450
IS_A2_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.36
Output dim: 0, lower bound: -0.5500877, upper bound: 0.5569198
IS_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.36
Output dim: 0, lower bound: -0.5420289, upper bound: 0.5251611
IS_A2_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 5.36
Output dim: 0, lower bound: -0.5383014, upper bound: 0.5569198
Binary search (step 3): status=Status.VERIFIED, low=0.0479713, high=0.0582672, mid=0.0479713, abs_max=0.6789970397949219
rel_dist={0: [-0.5828775183177297, 0.5828775183177297]}

## Binary search (step 4) starts
Candidate diff: 0.0531193


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5831267, upper bound: 0.5578402
time: 0.34 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5822902, upper bound: 0.5822902
time: 0.39 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.87 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.87
Output dim: 0, lower bound: -0.5831267, upper bound: 0.5578402
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.87
Output dim: 0, lower bound: -0.5822902, upper bound: 0.5822902

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.0674889, 0.4535027, -0.0306689, 0.6200731, -0.5525842, 0.4841716
1: 0.0201817, 0.5976362, -0.0918387, 0.8139617, -0.7937801, 0.6894749
2: -0.0460193, 0.5242411, -0.1828780, 0.6888012, -0.7348205, 0.7071191
3: -0.1192396, 0.5392005, -0.2733614, 0.7912076, -0.9104471, 0.8125620
4: -0.1227688, 0.6923177, -0.3010429, 0.9117734, -1.0345422, 0.9933606

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5578402, upper bound: 0.5578402
time: 0.34 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5578402, upper bound: 0.5578402
time: 0.36 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0106674, 0.5881721, -0.0365533, 0.6424438, -0.6531112, 0.6247254
1: -0.0680232, 0.7645921, -0.0992057, 0.8423603, -0.9103835, 0.8637978
2: -0.1492940, 0.6635130, -0.1934414, 0.7124153, -0.8617094, 0.8569544
3: -0.2353030, 0.7381678, -0.2859350, 0.8194001, -1.0547031, 1.0241028
4: -0.2566378, 0.8659467, -0.3152393, 0.9457331, -1.2023709, 1.1811860

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5578402, upper bound: 0.5822902
time: 0.34 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5578402, upper bound: 0.5822902
time: 0.35 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.44 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 2.44
Output dim: 0, lower bound: -0.5578402, upper bound: 0.5578402
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 2.44
Output dim: 0, lower bound: -0.5578402, upper bound: 0.5578402
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.44
Output dim: 0, lower bound: -0.5578402, upper bound: 0.5822902
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.44
Output dim: 0, lower bound: -0.5578402, upper bound: 0.5822902

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0106674, 0.5881721, 0.0674889, 0.4535027, -0.4641702, 0.5206832
1: -0.0680232, 0.7645921, 0.0201817, 0.5976362, -0.6656594, 0.7444105
2: -0.1492940, 0.6635130, -0.0460193, 0.5242411, -0.6735351, 0.7095323
3: -0.2353030, 0.7381678, -0.1192396, 0.5392005, -0.7745036, 0.8574073
4: -0.2566378, 0.8659467, -0.1227688, 0.6923177, -0.9489555, 0.9887154

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5450628, upper bound: 0.5633904
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5560570, upper bound: 0.5806276
time: 0.39 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0106674, 0.5881721, -0.0106674, 0.5881721, -0.5988396, 0.5988396
1: -0.0680232, 0.7645921, -0.0680232, 0.7645921, -0.8326153, 0.8326153
2: -0.1492940, 0.6635130, -0.1492940, 0.6635130, -0.8128070, 0.8128070
3: -0.2353030, 0.7381678, -0.2353030, 0.7381678, -0.9734708, 0.9734708
4: -0.2566378, 0.8659467, -0.2566378, 0.8659467, -1.1225845, 1.1225845

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5450628, upper bound: 0.5633905
time: 0.39 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5560570, upper bound: 0.5806276
time: 0.36 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.44 seconds
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 2.44
Output dim: 0, lower bound: -0.5450628, upper bound: 0.5633904
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.44
Output dim: 0, lower bound: -0.5560570, upper bound: 0.5806276
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 2.44
Output dim: 0, lower bound: -0.5450628, upper bound: 0.5633905
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.44
Output dim: 0, lower bound: -0.5560570, upper bound: 0.5806276

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0083361, 0.5719668, 0.0674889, 0.4535027, -0.4451666, 0.5044779
1: -0.0506135, 0.7433118, 0.0201817, 0.5976362, -0.6482497, 0.7231302
2: -0.1297052, 0.6455543, -0.0460193, 0.5242411, -0.6539463, 0.6915736
3: -0.2118729, 0.7126579, -0.1192396, 0.5392005, -0.7510734, 0.8318975
4: -0.2322460, 0.8428500, -0.1227688, 0.6923177, -0.9245638, 0.9656187

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5552654, upper bound: 0.5768399
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5552654, upper bound: 0.5768399
time: 0.35 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0083361, 0.5719668, -0.0106674, 0.5881721, -0.5798361, 0.5826342
1: -0.0506135, 0.7433118, -0.0680232, 0.7645921, -0.8152056, 0.8113350
2: -0.1297052, 0.6455543, -0.1492940, 0.6635130, -0.7932182, 0.7948483
3: -0.2118729, 0.7126579, -0.2353030, 0.7381678, -0.9500406, 0.9479610
4: -0.2322460, 0.8428500, -0.2566378, 0.8659467, -1.0981927, 1.0994878

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5555224, upper bound: 0.5616156
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5555224, upper bound: 0.5616156
time: 0.38 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.47 seconds
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -0.5552654, upper bound: 0.5768399
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -0.5552654, upper bound: 0.5768399
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 2.47
Output dim: 0, lower bound: -0.5555224, upper bound: 0.5616156
IS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 2.47
Output dim: 0, lower bound: -0.5555224, upper bound: 0.5616156

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0083361, 0.5719668, 0.0667267, 0.5276489, -0.5193129, 0.5052401
1: -0.0506135, 0.7433118, 0.0209925, 0.7202063, -0.7708198, 0.7223193
2: -0.1297052, 0.6455543, -0.0676523, 0.5800259, -0.7097311, 0.7132066
3: -0.2118729, 0.7126579, -0.1276886, 0.6454976, -0.8573705, 0.8403466
4: -0.2322460, 0.8428500, -0.1461909, 0.8046894, -1.0369354, 0.9890409

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4765382, upper bound: 0.5455130
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4756011, upper bound: 0.5711738
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 36

Time for candidate selection: 3.08 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5532468, upper bound: 0.5705621
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5426809, upper bound: 0.5713194
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0083361, 0.5719668, 0.0789177, 0.4454548, -0.4371188, 0.4930490
1: -0.0506135, 0.7433118, 0.0311321, 0.5851125, -0.6357260, 0.7121797
2: -0.1297052, 0.6455543, -0.0367235, 0.5172547, -0.6469599, 0.6822778
3: -0.2118729, 0.7126579, -0.1111584, 0.5267107, -0.7385836, 0.8238163
4: -0.2322460, 0.8428500, -0.1149590, 0.6824017, -0.9146477, 0.9578090

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4765383, upper bound: 0.5455130
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4756012, upper bound: 0.5711754
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 36

Time for candidate selection: 3.06 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5532469, upper bound: 0.5705621
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5451526, upper bound: 0.5713194
time: 0.43 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 5.52 seconds
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.52
Output dim: 0, lower bound: -0.5532468, upper bound: 0.5705621
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.52
Output dim: 0, lower bound: -0.5426809, upper bound: 0.5713194
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.52
Output dim: 0, lower bound: -0.5532469, upper bound: 0.5705621
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.52
Output dim: 0, lower bound: -0.5451526, upper bound: 0.5713194

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0274241, 0.5315738, 0.0667267, 0.5276489, -0.5002248, 0.4648471
1: -0.0291177, 0.6686592, 0.0209925, 0.7202063, -0.7493240, 0.6476666
2: -0.0887789, 0.6183543, -0.0676523, 0.5800259, -0.6688048, 0.6860067
3: -0.1754975, 0.6463004, -0.1276886, 0.6454976, -0.8209951, 0.7739891
4: -0.1838146, 0.7758477, -0.1461909, 0.8046894, -0.9885041, 0.9220386

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5457117, upper bound: 0.5506191
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5078514, upper bound: 0.5437989
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 3.17 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5491473, upper bound: 0.5263142
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5487440, upper bound: 0.5667979
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0149566, 0.5658405, 0.0667267, 0.5276489, -0.5126923, 0.4991139
1: -0.0463459, 0.7114239, 0.0209925, 0.7202063, -0.7665523, 0.6904314
2: -0.1064991, 0.6583920, -0.0676523, 0.5800259, -0.6865250, 0.7260443
3: -0.2061497, 0.6969915, -0.1276886, 0.6454976, -0.8516473, 0.8246801
4: -0.2113429, 0.8311484, -0.1461909, 0.8046894, -1.0160323, 0.9773393

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5047877, upper bound: 0.5589988
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.80 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5427440, upper bound: 0.5269680
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5672589
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0274241, 0.5315738, 0.0789177, 0.4454548, -0.4180307, 0.4526560
1: -0.0291177, 0.6686592, 0.0311321, 0.5851125, -0.6142302, 0.6375270
2: -0.0887789, 0.6183543, -0.0367235, 0.5172547, -0.6060336, 0.6550778
3: -0.1754975, 0.6463004, -0.1111584, 0.5267107, -0.7022082, 0.7574588
4: -0.1838146, 0.7758477, -0.1149590, 0.6824017, -0.8662163, 0.8908067

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5521510, upper bound: 0.5518962
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5067000, upper bound: 0.5398515
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 3.26 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5501535, upper bound: 0.5247910
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5508200, upper bound: 0.5573291
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0149566, 0.5658405, 0.0789177, 0.4454548, -0.4304982, 0.4869228
1: -0.0463459, 0.7114239, 0.0311321, 0.5851125, -0.6314585, 0.6802918
2: -0.1064991, 0.6583920, -0.0367235, 0.5172547, -0.6237538, 0.6951154
3: -0.2061497, 0.6969915, -0.1111584, 0.5267107, -0.7328604, 0.8081499
4: -0.2113429, 0.8311484, -0.1149590, 0.6824017, -0.8937446, 0.9461074

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5036363, upper bound: 0.5550514
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.78 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5421894, upper bound: 0.5254448
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5408537, upper bound: 0.5573291
time: 0.37 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 5.18 seconds
IS_A2_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.18
Output dim: 0, lower bound: -0.5491473, upper bound: 0.5263142
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -0.5487440, upper bound: 0.5667979
IS_A2_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.18
Output dim: 0, lower bound: -0.5427440, upper bound: 0.5269680
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5672589
IS_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.18
Output dim: 0, lower bound: -0.5501535, upper bound: 0.5247910
IS_A2_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.18
Output dim: 0, lower bound: -0.5508200, upper bound: 0.5573291
IS_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.18
Output dim: 0, lower bound: -0.5421894, upper bound: 0.5254448
IS_A2_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 5.18
Output dim: 0, lower bound: -0.5408537, upper bound: 0.5573291

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0274241, 0.5315738, 0.0677700, 0.5231507, -0.4957266, 0.4638038
1: -0.0291177, 0.6686592, 0.0225216, 0.7127589, -0.7418766, 0.6461375
2: -0.0887789, 0.6183543, -0.0648993, 0.5771931, -0.6659720, 0.6832536
3: -0.1754975, 0.6463004, -0.1257259, 0.6387789, -0.8142764, 0.7720263
4: -0.1838146, 0.7758477, -0.1438800, 0.7990351, -0.9828497, 0.9197277

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5223672, upper bound: 0.5659891
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5115022, upper bound: 0.5663484
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 36

Time for candidate selection: 3.15 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5487440, upper bound: 0.5667979
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5667979
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0149566, 0.5658405, 0.0677700, 0.5231507, -0.5081941, 0.4980705
1: -0.0463459, 0.7114239, 0.0225216, 0.7127589, -0.7591048, 0.6889023
2: -0.1064991, 0.6583920, -0.0648993, 0.5771931, -0.6836922, 0.7232913
3: -0.2061497, 0.6969915, -0.1257259, 0.6387789, -0.8449286, 0.8227174
4: -0.2113429, 0.8311484, -0.1438800, 0.7990351, -1.0103780, 0.9750284

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5174284, upper bound: 0.5661785
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5066713, upper bound: 0.5666426
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24

Time for candidate selection: 3.17 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5667979
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5672589
time: 0.36 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 5.51 seconds
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -0.5487440, upper bound: 0.5667979
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5667979
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5667979
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5672589

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0274241, 0.5315738, 0.0677700, 0.5231507, -0.4957266, 0.4638038
1: -0.0291177, 0.6686592, 0.0225216, 0.7127589, -0.7418766, 0.6461375
2: -0.0887789, 0.6183543, -0.0648993, 0.5771931, -0.6659720, 0.6832536
3: -0.1754975, 0.6463004, -0.1257259, 0.6387789, -0.8142764, 0.7720263
4: -0.1838146, 0.7758477, -0.1438800, 0.7990351, -0.9828497, 0.9197277

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5416673, upper bound: 0.5475433
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.86 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5487440, upper bound: 0.5263142
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5487440, upper bound: 0.5667979
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0151531, 0.5654141, 0.0677700, 0.5231507, -0.5079977, 0.4976441
1: -0.0459636, 0.7107825, 0.0225216, 0.7127589, -0.7587225, 0.6882609
2: -0.1062381, 0.6580169, -0.0648993, 0.5771931, -0.6834313, 0.7229162
3: -0.2055742, 0.6961249, -0.1257259, 0.6387789, -0.8443531, 0.8218507
4: -0.2109468, 0.8306049, -0.1438800, 0.7990351, -1.0099819, 0.9744849

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.58 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5263142
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5667979
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0274241, 0.5315738, 0.0677700, 0.5231507, -0.4957266, 0.4638038
1: -0.0291177, 0.6686592, 0.0225216, 0.7127589, -0.7418766, 0.6461375
2: -0.0887789, 0.6183543, -0.0648993, 0.5771931, -0.6659720, 0.6832536
3: -0.1754975, 0.6463004, -0.1257259, 0.6387789, -0.8142764, 0.7720263
4: -0.1838146, 0.7758477, -0.1438800, 0.7990351, -0.9828497, 0.9197277

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5416673, upper bound: 0.5475433
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.95 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5487440, upper bound: 0.5263142
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5487440, upper bound: 0.5667979
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0149566, 0.5658405, 0.0677700, 0.5231507, -0.5081941, 0.4980705
1: -0.0463459, 0.7114239, 0.0225216, 0.7127589, -0.7591048, 0.6889023
2: -0.1064991, 0.6583920, -0.0648993, 0.5771931, -0.6836922, 0.7232913
3: -0.2061497, 0.6969915, -0.1257259, 0.6387789, -0.8449286, 0.8227174
4: -0.2113429, 0.8311484, -0.1438800, 0.7990351, -1.0103780, 0.9750284

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.60 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5269680
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5672589
time: 0.38 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 5.08 seconds
IS_A2_B1_A2_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 5.08
Output dim: 0, lower bound: -0.5487440, upper bound: 0.5263142
IS_A2_B1_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.08
Output dim: 0, lower bound: -0.5487440, upper bound: 0.5667979
IS_A2_B1_A2_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 5.08
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5263142
IS_A2_B1_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.08
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5667979
IS_A2_B1_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 5.08
Output dim: 0, lower bound: -0.5487440, upper bound: 0.5263142
IS_A2_B1_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.08
Output dim: 0, lower bound: -0.5487440, upper bound: 0.5667979
IS_A2_B1_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 5.08
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5269680
IS_A2_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.08
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5672589

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0274241, 0.5315738, 0.0677700, 0.5231507, -0.4957266, 0.4638038
1: -0.0291177, 0.6686592, 0.0225216, 0.7127589, -0.7418766, 0.6461375
2: -0.0887789, 0.6183543, -0.0648993, 0.5771931, -0.6659720, 0.6832536
3: -0.1754975, 0.6463004, -0.1257259, 0.6387789, -0.8142764, 0.7720263
4: -0.1838146, 0.7758477, -0.1438800, 0.7990351, -0.9828497, 0.9197277

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5223672, upper bound: 0.5659891
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5115022, upper bound: 0.5663484
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 36

Time for candidate selection: 3.17 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5487440, upper bound: 0.5667979
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5667979
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0151531, 0.5654141, 0.0677700, 0.5231507, -0.5079977, 0.4976441
1: -0.0459636, 0.7107825, 0.0225216, 0.7127589, -0.7587225, 0.6882609
2: -0.1062381, 0.6580169, -0.0648993, 0.5771931, -0.6834313, 0.7229162
3: -0.2055742, 0.6961249, -0.1257259, 0.6387789, -0.8443531, 0.8218507
4: -0.2109468, 0.8306049, -0.1438800, 0.7990351, -1.0099819, 0.9744849

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5174284, upper bound: 0.5659891
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5066713, upper bound: 0.5663484
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24

Time for candidate selection: 3.22 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5667979
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5667979
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0274241, 0.5315738, 0.0677700, 0.5231507, -0.4957266, 0.4638038
1: -0.0291177, 0.6686592, 0.0225216, 0.7127589, -0.7418766, 0.6461375
2: -0.0887789, 0.6183543, -0.0648993, 0.5771931, -0.6659720, 0.6832536
3: -0.1754975, 0.6463004, -0.1257259, 0.6387789, -0.8142764, 0.7720263
4: -0.1838146, 0.7758477, -0.1438800, 0.7990351, -0.9828497, 0.9197277

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5223672, upper bound: 0.5659891
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5115022, upper bound: 0.5663484
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 36

Time for candidate selection: 3.31 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5487440, upper bound: 0.5667979
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5667979
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0149566, 0.5658405, 0.0677700, 0.5231507, -0.5081941, 0.4980705
1: -0.0463459, 0.7114239, 0.0225216, 0.7127589, -0.7591048, 0.6889023
2: -0.1064991, 0.6583920, -0.0648993, 0.5771931, -0.6836922, 0.7232913
3: -0.2061497, 0.6969915, -0.1257259, 0.6387789, -0.8449286, 0.8227174
4: -0.2113429, 0.8311484, -0.1438800, 0.7990351, -1.0103780, 0.9750284

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5174284, upper bound: 0.5661785
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5066713, upper bound: 0.5666426
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24

Time for candidate selection: 3.32 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5667979
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5672589
time: 0.38 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 5.73 seconds
IS_A2_B1_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 5.73
Output dim: 0, lower bound: -0.5487440, upper bound: 0.5667979
IS_A2_B1_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 5.73
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5667979
IS_A2_B1_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 5.73
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5667979
IS_A2_B1_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 5.73
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5667979
IS_A2_B1_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 5.73
Output dim: 0, lower bound: -0.5487440, upper bound: 0.5667979
IS_A2_B1_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 5.73
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5667979
IS_A2_B1_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 5.73
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5667979
IS_A2_B1_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 5.73
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5672589

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0274241, 0.5315738, 0.0677700, 0.5231507, -0.4957266, 0.4638038
1: -0.0291177, 0.6686592, 0.0225216, 0.7127589, -0.7418766, 0.6461375
2: -0.0887789, 0.6183543, -0.0648993, 0.5771931, -0.6659720, 0.6832536
3: -0.1754975, 0.6463004, -0.1257259, 0.6387789, -0.8142764, 0.7720263
4: -0.1838146, 0.7758477, -0.1438800, 0.7990351, -0.9828497, 0.9197277

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5416673, upper bound: 0.5475433
time: 0.44 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.97 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5487440, upper bound: 0.5263142
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5487440, upper bound: 0.5667979
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0151531, 0.5654141, 0.0677700, 0.5231507, -0.5079977, 0.4976441
1: -0.0459636, 0.7107825, 0.0225216, 0.7127589, -0.7587225, 0.6882609
2: -0.1062381, 0.6580169, -0.0648993, 0.5771931, -0.6834313, 0.7229162
3: -0.2055742, 0.6961249, -0.1257259, 0.6387789, -0.8443531, 0.8218507
4: -0.2109468, 0.8306049, -0.1438800, 0.7990351, -1.0099819, 0.9744849

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.61 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5263142
time: 0.48 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5667979
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0274241, 0.5315738, 0.0677700, 0.5231507, -0.4957266, 0.4638038
1: -0.0291177, 0.6686592, 0.0225216, 0.7127589, -0.7418766, 0.6461375
2: -0.0887789, 0.6183543, -0.0648993, 0.5771931, -0.6659720, 0.6832536
3: -0.1754975, 0.6463004, -0.1257259, 0.6387789, -0.8142764, 0.7720263
4: -0.1838146, 0.7758477, -0.1438800, 0.7990351, -0.9828497, 0.9197277

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5416673, upper bound: 0.5475433
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 3.01 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5487440, upper bound: 0.5263142
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5487440, upper bound: 0.5667979
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0151531, 0.5654141, 0.0677700, 0.5231507, -0.5079977, 0.4976441
1: -0.0459636, 0.7107825, 0.0225216, 0.7127589, -0.7587225, 0.6882609
2: -0.1062381, 0.6580169, -0.0648993, 0.5771931, -0.6834313, 0.7229162
3: -0.2055742, 0.6961249, -0.1257259, 0.6387789, -0.8443531, 0.8218507
4: -0.2109468, 0.8306049, -0.1438800, 0.7990351, -1.0099819, 0.9744849

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.63 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5263142
time: 0.47 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5667979
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0274241, 0.5315738, 0.0677700, 0.5231507, -0.4957266, 0.4638038
1: -0.0291177, 0.6686592, 0.0225216, 0.7127589, -0.7418766, 0.6461375
2: -0.0887789, 0.6183543, -0.0648993, 0.5771931, -0.6659720, 0.6832536
3: -0.1754975, 0.6463004, -0.1257259, 0.6387789, -0.8142764, 0.7720263
4: -0.1838146, 0.7758477, -0.1438800, 0.7990351, -0.9828497, 0.9197277

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5416673, upper bound: 0.5475433
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 3.02 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5487440, upper bound: 0.5263142
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5487440, upper bound: 0.5667979
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0151531, 0.5654141, 0.0677700, 0.5231507, -0.5079977, 0.4976441
1: -0.0459636, 0.7107825, 0.0225216, 0.7127589, -0.7587225, 0.6882609
2: -0.1062381, 0.6580169, -0.0648993, 0.5771931, -0.6834313, 0.7229162
3: -0.2055742, 0.6961249, -0.1257259, 0.6387789, -0.8443531, 0.8218507
4: -0.2109468, 0.8306049, -0.1438800, 0.7990351, -1.0099819, 0.9744849

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.66 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5263142
time: 0.47 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5667979
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0274241, 0.5315738, 0.0677700, 0.5231507, -0.4957266, 0.4638038
1: -0.0291177, 0.6686592, 0.0225216, 0.7127589, -0.7418766, 0.6461375
2: -0.0887789, 0.6183543, -0.0648993, 0.5771931, -0.6659720, 0.6832536
3: -0.1754975, 0.6463004, -0.1257259, 0.6387789, -0.8142764, 0.7720263
4: -0.1838146, 0.7758477, -0.1438800, 0.7990351, -0.9828497, 0.9197277

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5416673, upper bound: 0.5475433
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 3.05 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5487440, upper bound: 0.5263142
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5487440, upper bound: 0.5667979
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0149566, 0.5658405, 0.0677700, 0.5231507, -0.5081941, 0.4980705
1: -0.0463459, 0.7114239, 0.0225216, 0.7127589, -0.7591048, 0.6889023
2: -0.1064991, 0.6583920, -0.0648993, 0.5771931, -0.6836922, 0.7232913
3: -0.2061497, 0.6969915, -0.1257259, 0.6387789, -0.8449286, 0.8227174
4: -0.2113429, 0.8311484, -0.1438800, 0.7990351, -1.0103780, 0.9750284

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.69 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5269680
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5672589
time: 0.41 seconds

## Summary of splitting at layer (split count: 9)
- Time for IS candidates: 5.22 seconds
IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 10, time: 5.22
Output dim: 0, lower bound: -0.5487440, upper bound: 0.5263142
IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 5.22
Output dim: 0, lower bound: -0.5487440, upper bound: 0.5667979
IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 5.22
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5263142
IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 5.22
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5667979
IS_A2_B1_A2_B1_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 10, time: 5.22
Output dim: 0, lower bound: -0.5487440, upper bound: 0.5263142
IS_A2_B1_A2_B1_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 5.22
Output dim: 0, lower bound: -0.5487440, upper bound: 0.5667979
IS_A2_B1_A2_B1_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 5.22
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5263142
IS_A2_B1_A2_B1_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 5.22
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5667979
IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 10, time: 5.22
Output dim: 0, lower bound: -0.5487440, upper bound: 0.5263142
IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 5.22
Output dim: 0, lower bound: -0.5487440, upper bound: 0.5667979
IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 5.22
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5263142
IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 5.22
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5667979
IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 10, time: 5.22
Output dim: 0, lower bound: -0.5487440, upper bound: 0.5263142
IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 5.22
Output dim: 0, lower bound: -0.5487440, upper bound: 0.5667979
IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 5.22
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5269680
IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 5.22
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5672589

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0274241, 0.5315738, 0.0677700, 0.5231507, -0.4957266, 0.4638038
1: -0.0291177, 0.6686592, 0.0225216, 0.7127589, -0.7418766, 0.6461375
2: -0.0887789, 0.6183543, -0.0648993, 0.5771931, -0.6659720, 0.6832536
3: -0.1754975, 0.6463004, -0.1257259, 0.6387789, -0.8142764, 0.7720263
4: -0.1838146, 0.7758477, -0.1438800, 0.7990351, -0.9828497, 0.9197277

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5223672, upper bound: 0.5659891
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5115022, upper bound: 0.5663484
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 36

Time for candidate selection: 3.33 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5487440, upper bound: 0.5667979
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5667979
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0151531, 0.5654141, 0.0677700, 0.5231507, -0.5079977, 0.4976441
1: -0.0459636, 0.7107825, 0.0225216, 0.7127589, -0.7587225, 0.6882609
2: -0.1062381, 0.6580169, -0.0648993, 0.5771931, -0.6834313, 0.7229162
3: -0.2055742, 0.6961249, -0.1257259, 0.6387789, -0.8443531, 0.8218507
4: -0.2109468, 0.8306049, -0.1438800, 0.7990351, -1.0099819, 0.9744849

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5174284, upper bound: 0.5659891
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5066713, upper bound: 0.5663484
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24

Time for candidate selection: 3.35 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5667979
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5667979
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0274241, 0.5315738, 0.0677700, 0.5231507, -0.4957266, 0.4638038
1: -0.0291177, 0.6686592, 0.0225216, 0.7127589, -0.7418766, 0.6461375
2: -0.0887789, 0.6183543, -0.0648993, 0.5771931, -0.6659720, 0.6832536
3: -0.1754975, 0.6463004, -0.1257259, 0.6387789, -0.8142764, 0.7720263
4: -0.1838146, 0.7758477, -0.1438800, 0.7990351, -0.9828497, 0.9197277

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5223672, upper bound: 0.5659891
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5115022, upper bound: 0.5663484
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 36

Time for candidate selection: 3.36 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5487440, upper bound: 0.5667979
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5667979
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0151531, 0.5654141, 0.0677700, 0.5231507, -0.5079977, 0.4976441
1: -0.0459636, 0.7107825, 0.0225216, 0.7127589, -0.7587225, 0.6882609
2: -0.1062381, 0.6580169, -0.0648993, 0.5771931, -0.6834313, 0.7229162
3: -0.2055742, 0.6961249, -0.1257259, 0.6387789, -0.8443531, 0.8218507
4: -0.2109468, 0.8306049, -0.1438800, 0.7990351, -1.0099819, 0.9744849

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5174284, upper bound: 0.5659891
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5066713, upper bound: 0.5663484
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24

Time for candidate selection: 3.38 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5667979
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5667979
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0274241, 0.5315738, 0.0677700, 0.5231507, -0.4957266, 0.4638038
1: -0.0291177, 0.6686592, 0.0225216, 0.7127589, -0.7418766, 0.6461375
2: -0.0887789, 0.6183543, -0.0648993, 0.5771931, -0.6659720, 0.6832536
3: -0.1754975, 0.6463004, -0.1257259, 0.6387789, -0.8142764, 0.7720263
4: -0.1838146, 0.7758477, -0.1438800, 0.7990351, -0.9828497, 0.9197277

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5223672, upper bound: 0.5659891
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5115022, upper bound: 0.5663484
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 36

Time for candidate selection: 3.36 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5487440, upper bound: 0.5667979
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5667979
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0151531, 0.5654141, 0.0677700, 0.5231507, -0.5079977, 0.4976441
1: -0.0459636, 0.7107825, 0.0225216, 0.7127589, -0.7587225, 0.6882609
2: -0.1062381, 0.6580169, -0.0648993, 0.5771931, -0.6834313, 0.7229162
3: -0.2055742, 0.6961249, -0.1257259, 0.6387789, -0.8443531, 0.8218507
4: -0.2109468, 0.8306049, -0.1438800, 0.7990351, -1.0099819, 0.9744849

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5174284, upper bound: 0.5659891
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5066713, upper bound: 0.5663484
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24

Time for candidate selection: 3.33 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5667979
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5667979
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0274241, 0.5315738, 0.0677700, 0.5231507, -0.4957266, 0.4638038
1: -0.0291177, 0.6686592, 0.0225216, 0.7127589, -0.7418766, 0.6461375
2: -0.0887789, 0.6183543, -0.0648993, 0.5771931, -0.6659720, 0.6832536
3: -0.1754975, 0.6463004, -0.1257259, 0.6387789, -0.8142764, 0.7720263
4: -0.1838146, 0.7758477, -0.1438800, 0.7990351, -0.9828497, 0.9197277

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5223672, upper bound: 0.5659891
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5115022, upper bound: 0.5663484
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 36

Time for candidate selection: 3.40 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5487440, upper bound: 0.5667979
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5667979
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0149566, 0.5658405, 0.0677700, 0.5231507, -0.5081941, 0.4980705
1: -0.0463459, 0.7114239, 0.0225216, 0.7127589, -0.7591048, 0.6889023
2: -0.1064991, 0.6583920, -0.0648993, 0.5771931, -0.6836922, 0.7232913
3: -0.2061497, 0.6969915, -0.1257259, 0.6387789, -0.8449286, 0.8227174
4: -0.2113429, 0.8311484, -0.1438800, 0.7990351, -1.0103780, 0.9750284

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5174284, upper bound: 0.5661785
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5066713, upper bound: 0.5666426
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24

Time for candidate selection: 3.39 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5667979
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5672589
time: 0.40 seconds

## Summary of splitting at layer (split count: 10)
- Time for IS candidates: 5.95 seconds
IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 11, time: 5.95
Output dim: 0, lower bound: -0.5487440, upper bound: 0.5667979
IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 11, time: 5.95
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5667979
IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 11, time: 5.95
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5667979
IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 5.95
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5667979
IS_A2_B1_A2_B1_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 11, time: 5.95
Output dim: 0, lower bound: -0.5487440, upper bound: 0.5667979
IS_A2_B1_A2_B1_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 11, time: 5.95
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5667979
IS_A2_B1_A2_B1_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 11, time: 5.95
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5667979
IS_A2_B1_A2_B1_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 5.95
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5667979
IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 11, time: 5.95
Output dim: 0, lower bound: -0.5487440, upper bound: 0.5667979
IS_A2_B1_A2_B1_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 11, time: 5.95
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5667979
IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 11, time: 5.95
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5667979
IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 5.95
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5667979
IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 11, time: 5.95
Output dim: 0, lower bound: -0.5487440, upper bound: 0.5667979
IS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 11, time: 5.95
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5667979
IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 11, time: 5.95
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5667979
IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 5.95
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5672589

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0274241, 0.5315738, 0.0677700, 0.5231507, -0.4957266, 0.4638038
1: -0.0291177, 0.6686592, 0.0225216, 0.7127589, -0.7418766, 0.6461375
2: -0.0887789, 0.6183543, -0.0648993, 0.5771931, -0.6659720, 0.6832536
3: -0.1754975, 0.6463004, -0.1257259, 0.6387789, -0.8142764, 0.7720263
4: -0.1838146, 0.7758477, -0.1438800, 0.7990351, -0.9828497, 0.9197277

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5416673, upper bound: 0.5475433
time: 0.44 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 3.08 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5487440, upper bound: 0.5263142
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5487440, upper bound: 0.5667979
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0151531, 0.5654141, 0.0677700, 0.5231507, -0.5079977, 0.4976441
1: -0.0459636, 0.7107825, 0.0225216, 0.7127589, -0.7587225, 0.6882609
2: -0.1062381, 0.6580169, -0.0648993, 0.5771931, -0.6834313, 0.7229162
3: -0.2055742, 0.6961249, -0.1257259, 0.6387789, -0.8443531, 0.8218507
4: -0.2109468, 0.8306049, -0.1438800, 0.7990351, -1.0099819, 0.9744849

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.67 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5263142
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5667979
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0274241, 0.5315738, 0.0677700, 0.5231507, -0.4957266, 0.4638038
1: -0.0291177, 0.6686592, 0.0225216, 0.7127589, -0.7418766, 0.6461375
2: -0.0887789, 0.6183543, -0.0648993, 0.5771931, -0.6659720, 0.6832536
3: -0.1754975, 0.6463004, -0.1257259, 0.6387789, -0.8142764, 0.7720263
4: -0.1838146, 0.7758477, -0.1438800, 0.7990351, -0.9828497, 0.9197277

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5416673, upper bound: 0.5475433
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 3.07 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5487440, upper bound: 0.5263142
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5487440, upper bound: 0.5667979
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0151531, 0.5654141, 0.0677700, 0.5231507, -0.5079977, 0.4976441
1: -0.0459636, 0.7107825, 0.0225216, 0.7127589, -0.7587225, 0.6882609
2: -0.1062381, 0.6580169, -0.0648993, 0.5771931, -0.6834313, 0.7229162
3: -0.2055742, 0.6961249, -0.1257259, 0.6387789, -0.8443531, 0.8218507
4: -0.2109468, 0.8306049, -0.1438800, 0.7990351, -1.0099819, 0.9744849

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 5

Time for candidate selection: 2.72 seconds

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5263142
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5408172, upper bound: 0.5667979
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0274241, 0.5315738, 0.0677700, 0.5231507, -0.4957266, 0.4638038
1: -0.0291177, 0.6686592, 0.0225216, 0.7127589, -0.7418766, 0.6461375
2: -0.0887789, 0.6183543, -0.0648993, 0.5771931, -0.6659720, 0.6832536
3: -0.1754975, 0.6463004, -0.1257259, 0.6387789, -0.8142764, 0.7720263
4: -0.1838146, 0.7758477, -0.1438800, 0.7990351, -0.9828497, 0.9197277

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5416673, upper bound: 0.5475433
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 4): status=Status.UNKNOWN, low=0.0479713, high=0.0531193, mid=0.0531193, abs_max=0.6789970397949219
rel_dist={0: [-0.5846828489836718, 0.5846828489836726]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.04797133675403131
execution time: 1149.50 seconds
