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
execution time: IAR + LP analysis = 1.57 + 1.02 = 2.59 seconds
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
Binary search time: 45.75 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.017083602027241795


# Relational Split (RS_dual_Z) starts
Time budget: 1151.67 seconds

## Binary search (step 0) starts
Candidate diff: 0.0994509


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5854361, upper bound: 0.5881033
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5881033, upper bound: 0.5854361
time: 0.36 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.86 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.86
Output dim: 0, lower bound: -0.5854361, upper bound: 0.5881033
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.86
Output dim: 0, lower bound: -0.5881033, upper bound: 0.5854361

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5819236, upper bound: 0.5832535
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5681019, upper bound: 0.5858474
time: 0.32 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5858474, upper bound: 0.5681019
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5832535, upper bound: 0.5819236
time: 0.35 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.23 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.23
Output dim: 0, lower bound: -0.5819236, upper bound: 0.5832535
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.23
Output dim: 0, lower bound: -0.5681019, upper bound: 0.5858474
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.23
Output dim: 0, lower bound: -0.5858474, upper bound: 0.5681019
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.23
Output dim: 0, lower bound: -0.5832535, upper bound: 0.5819236

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5732267, upper bound: 0.5555948
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5422272, upper bound: 0.5827478
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5639934, upper bound: 0.5442548
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5441623, upper bound: 0.5828222
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5828222, upper bound: 0.5441623
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5442548, upper bound: 0.5639934
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5827478, upper bound: 0.5422272
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5555948, upper bound: 0.5732267
time: 0.40 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.32 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.32
Output dim: 0, lower bound: -0.5732267, upper bound: 0.5555948
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.32
Output dim: 0, lower bound: -0.5422272, upper bound: 0.5827478
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.32
Output dim: 0, lower bound: -0.5639934, upper bound: 0.5442548
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.32
Output dim: 0, lower bound: -0.5441623, upper bound: 0.5828222
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.32
Output dim: 0, lower bound: -0.5828222, upper bound: 0.5441623
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.32
Output dim: 0, lower bound: -0.5442548, upper bound: 0.5639934
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.32
Output dim: 0, lower bound: -0.5827478, upper bound: 0.5422272
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.32
Output dim: 0, lower bound: -0.5555948, upper bound: 0.5732267

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5225306, upper bound: 0.5553310
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5731152, upper bound: 0.5552399
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5411046, upper bound: 0.5824656
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5419741, upper bound: 0.5691682
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5395691, upper bound: 0.5826954
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5439453, upper bound: 0.5719572
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5719572, upper bound: 0.5439453
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5826954, upper bound: 0.5395691
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5691682, upper bound: 0.5419741
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5824656, upper bound: 0.5411046
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5552399, upper bound: 0.5731152
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5553310, upper bound: 0.5439457
time: 0.34 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.65 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 0, lower bound: -0.5225306, upper bound: 0.5553310
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 0, lower bound: -0.5731152, upper bound: 0.5552399
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 0, lower bound: -0.5411046, upper bound: 0.5824656
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 0, lower bound: -0.5419741, upper bound: 0.5691682
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 0, lower bound: -0.5395691, upper bound: 0.5826954
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 0, lower bound: -0.5439453, upper bound: 0.5719572
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 0, lower bound: -0.5719572, upper bound: 0.5439453
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 0, lower bound: -0.5826954, upper bound: 0.5395691
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 0, lower bound: -0.5691682, upper bound: 0.5419741
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 0, lower bound: -0.5824656, upper bound: 0.5411046
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 0, lower bound: -0.5552399, upper bound: 0.5731152
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 0, lower bound: -0.5553310, upper bound: 0.5439457

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5650934, upper bound: 0.5224687
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5731104, upper bound: 0.5551714
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5410553, upper bound: 0.5737050
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5340679, upper bound: 0.5824606
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5419270, upper bound: 0.5224687
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5401907, upper bound: 0.5688342
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5395141, upper bound: 0.5780496
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5308092, upper bound: 0.5826907
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5438800, upper bound: 0.5272497
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5438159, upper bound: 0.5715214
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5715214, upper bound: 0.5438159
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5272497, upper bound: 0.5438800
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5826907, upper bound: 0.5308092
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5780496, upper bound: 0.5395141
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5224687, upper bound: 0.5224687
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5224687, upper bound: 0.5419270
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5824606, upper bound: 0.5340679
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5737050, upper bound: 0.5410553
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5551714, upper bound: 0.5731104
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5224687, upper bound: 0.5650934
time: 0.34 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.35 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.35
Output dim: 0, lower bound: -0.5650934, upper bound: 0.5224687
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -0.5731104, upper bound: 0.5551714
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -0.5410553, upper bound: 0.5737050
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -0.5340679, upper bound: 0.5824606
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.35
Output dim: 0, lower bound: -0.5419270, upper bound: 0.5224687
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -0.5401907, upper bound: 0.5688342
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -0.5395141, upper bound: 0.5780496
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -0.5308092, upper bound: 0.5826907
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.35
Output dim: 0, lower bound: -0.5438800, upper bound: 0.5272497
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -0.5438159, upper bound: 0.5715214
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -0.5715214, upper bound: 0.5438159
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.35
Output dim: 0, lower bound: -0.5272497, upper bound: 0.5438800
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -0.5826907, upper bound: 0.5308092
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -0.5780496, upper bound: 0.5395141
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.35
Output dim: 0, lower bound: -0.5224687, upper bound: 0.5224687
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.35
Output dim: 0, lower bound: -0.5224687, upper bound: 0.5419270
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -0.5824606, upper bound: 0.5340679
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -0.5737050, upper bound: 0.5410553
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -0.5551714, upper bound: 0.5731104
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.35
Output dim: 0, lower bound: -0.5224687, upper bound: 0.5650934

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5634311, upper bound: 0.5365798
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5535169, upper bound: 0.5509407
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.26 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5339610, upper bound: 0.5590787
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203890, upper bound: 0.5677562
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.25 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5291521, upper bound: 0.5672077
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203789, upper bound: 0.5770248
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.27 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5337286, upper bound: 0.5507160
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5203878, upper bound: 0.5634436
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.28 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5331724, upper bound: 0.5633422
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203876, upper bound: 0.5712291
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.24 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5262855, upper bound: 0.5673497
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203719, upper bound: 0.5771027
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.33 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5373765, upper bound: 0.5542694
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5228855, upper bound: 0.5654198
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.26 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5654198, upper bound: 0.5228855
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5542694, upper bound: 0.5373765
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5771027, upper bound: 0.5203719
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5673497, upper bound: 0.5262855
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.25 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5712291, upper bound: 0.5203876
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5633422, upper bound: 0.5331724
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.27 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5770248, upper bound: 0.5203789
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5672077, upper bound: 0.5291521
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5677562, upper bound: 0.5203890
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5590787, upper bound: 0.5339610
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.31 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5509407, upper bound: 0.5535169
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5365798, upper bound: 0.5634311
time: 0.36 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.62 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.62
Output dim: 0, lower bound: -0.5634311, upper bound: 0.5365798
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.62
Output dim: 0, lower bound: -0.5535169, upper bound: 0.5509407
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.62
Output dim: 0, lower bound: -0.5339610, upper bound: 0.5590787
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.62
Output dim: 0, lower bound: -0.5203890, upper bound: 0.5677562
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.62
Output dim: 0, lower bound: -0.5291521, upper bound: 0.5672077
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.62
Output dim: 0, lower bound: -0.5203789, upper bound: 0.5770248
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.62
Output dim: 0, lower bound: -0.5337286, upper bound: 0.5507160
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.62
Output dim: 0, lower bound: -0.5203878, upper bound: 0.5634436
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.62
Output dim: 0, lower bound: -0.5331724, upper bound: 0.5633422
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.62
Output dim: 0, lower bound: -0.5203876, upper bound: 0.5712291
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.62
Output dim: 0, lower bound: -0.5262855, upper bound: 0.5673497
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.62
Output dim: 0, lower bound: -0.5203719, upper bound: 0.5771027
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.62
Output dim: 0, lower bound: -0.5373765, upper bound: 0.5542694
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.62
Output dim: 0, lower bound: -0.5228855, upper bound: 0.5654198
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.62
Output dim: 0, lower bound: -0.5654198, upper bound: 0.5228855
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.62
Output dim: 0, lower bound: -0.5542694, upper bound: 0.5373765
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.62
Output dim: 0, lower bound: -0.5771027, upper bound: 0.5203719
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.62
Output dim: 0, lower bound: -0.5673497, upper bound: 0.5262855
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.62
Output dim: 0, lower bound: -0.5712291, upper bound: 0.5203876
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.62
Output dim: 0, lower bound: -0.5633422, upper bound: 0.5331724
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.62
Output dim: 0, lower bound: -0.5770248, upper bound: 0.5203789
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.62
Output dim: 0, lower bound: -0.5672077, upper bound: 0.5291521
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.62
Output dim: 0, lower bound: -0.5677562, upper bound: 0.5203890
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.62
Output dim: 0, lower bound: -0.5590787, upper bound: 0.5339610
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.62
Output dim: 0, lower bound: -0.5509407, upper bound: 0.5535169
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.62
Output dim: 0, lower bound: -0.5365798, upper bound: 0.5634311

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5203640, upper bound: 0.5213139
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203276, upper bound: 0.5675479
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5291295, upper bound: 0.5224842
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5291006, upper bound: 0.5670215
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5203539, upper bound: 0.5256562
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203224, upper bound: 0.5768506
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5203625, upper bound: 0.5260726
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203268, upper bound: 0.5710205
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5262574, upper bound: 0.5264616
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5262423, upper bound: 0.5671637
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5203469, upper bound: 0.5296653
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203187, upper bound: 0.5769270
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5227134, upper bound: 0.5524678
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5204377, upper bound: 0.5653871
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5653871, upper bound: 0.5204377
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5524678, upper bound: 0.5227134
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5168306, upper bound: 0.5134462
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5168306, upper bound: 0.5134462
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5149852, upper bound: 0.5134462
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5149852, upper bound: 0.5134462
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5165941, upper bound: 0.5134637
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5165941, upper bound: 0.5134630
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5768506, upper bound: 0.5203224
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5256562, upper bound: 0.5203539
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5670215, upper bound: 0.5291006
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5224842, upper bound: 0.5291295
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5675479, upper bound: 0.5203276
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5213139, upper bound: 0.5203640
time: 0.34 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.44 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.5203640, upper bound: 0.5213139
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.5203276, upper bound: 0.5675479
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.5291295, upper bound: 0.5224842
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.5291006, upper bound: 0.5670215
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.5203539, upper bound: 0.5256562
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.5203224, upper bound: 0.5768506
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.5203625, upper bound: 0.5260726
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.5203268, upper bound: 0.5710205
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.5262574, upper bound: 0.5264616
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.5262423, upper bound: 0.5671637
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.5203469, upper bound: 0.5296653
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.5203187, upper bound: 0.5769270
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.5227134, upper bound: 0.5524678
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.5204377, upper bound: 0.5653871
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.5653871, upper bound: 0.5204377
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.5524678, upper bound: 0.5227134
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.5168306, upper bound: 0.5134462
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.5168306, upper bound: 0.5134462
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.5149852, upper bound: 0.5134462
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.5149852, upper bound: 0.5134462
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.5165941, upper bound: 0.5134637
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.5165941, upper bound: 0.5134630
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.5768506, upper bound: 0.5203224
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.5256562, upper bound: 0.5203539
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.5670215, upper bound: 0.5291006
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.5224842, upper bound: 0.5291295
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.5675479, upper bound: 0.5203276
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.5213139, upper bound: 0.5203640

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5138510
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5143041
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133644, upper bound: 0.5134549
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133983, upper bound: 0.5134549
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5158407
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5158407
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5165533
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5165533
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5149469
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5149469
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5167898
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5167898
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5157573
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5157573
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5157573, upper bound: 0.5133628
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5157573, upper bound: 0.5133628
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5158407, upper bound: 0.5133628
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5158407, upper bound: 0.5133628
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5134549, upper bound: 0.5133983
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5134549, upper bound: 0.5133644
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5143041, upper bound: 0.5133628
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5138510, upper bound: 0.5133628
time: 0.41 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 2.51 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.51
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5138510
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.51
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5143041
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.51
Output dim: 0, lower bound: -0.5133644, upper bound: 0.5134549
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.51
Output dim: 0, lower bound: -0.5133983, upper bound: 0.5134549
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.51
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5158407
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.51
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5158407
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.51
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5165533
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.51
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5165533
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.51
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5149469
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.51
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5149469
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.51
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5167898
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.51
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5167898
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.51
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5157573
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.51
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5157573
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.51
Output dim: 0, lower bound: -0.5157573, upper bound: 0.5133628
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.51
Output dim: 0, lower bound: -0.5157573, upper bound: 0.5133628
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.51
Output dim: 0, lower bound: -0.5158407, upper bound: 0.5133628
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.51
Output dim: 0, lower bound: -0.5158407, upper bound: 0.5133628
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.51
Output dim: 0, lower bound: -0.5134549, upper bound: 0.5133983
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.51
Output dim: 0, lower bound: -0.5134549, upper bound: 0.5133644
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.51
Output dim: 0, lower bound: -0.5143041, upper bound: 0.5133628
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.51
Output dim: 0, lower bound: -0.5138510, upper bound: 0.5133628
Binary search (step 0): status=Status.VERIFIED, low=0.0994509, high=0.1818182, mid=0.0994509, abs_max=0.6789970397949219
rel_dist={0: [-0.5950600062778149, 0.5950600062778155]}

## Binary search (step 1) starts
Candidate diff: 0.1406345


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5894953, upper bound: 0.5894953
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5894953, upper bound: 0.5894953
time: 0.36 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.87 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.87
Output dim: 0, lower bound: -0.5894953, upper bound: 0.5894953
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.87
Output dim: 0, lower bound: -0.5894953, upper bound: 0.5894953

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5868319, upper bound: 0.5832535
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5681019, upper bound: 0.5868319
time: 0.32 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5868319, upper bound: 0.5681019
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5832535, upper bound: 0.5868319
time: 0.36 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.32 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.32
Output dim: 0, lower bound: -0.5868319, upper bound: 0.5832535
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.32
Output dim: 0, lower bound: -0.5681019, upper bound: 0.5868319
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.32
Output dim: 0, lower bound: -0.5868319, upper bound: 0.5681019
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.32
Output dim: 0, lower bound: -0.5832535, upper bound: 0.5868319

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5732267, upper bound: 0.5555948
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5422272, upper bound: 0.5827478
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5639934, upper bound: 0.5442548
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5441623, upper bound: 0.5828222
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5828222, upper bound: 0.5441623
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5442548, upper bound: 0.5639934
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5827478, upper bound: 0.5422272
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5555948, upper bound: 0.5732267
time: 0.37 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.42 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.42
Output dim: 0, lower bound: -0.5732267, upper bound: 0.5555948
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.42
Output dim: 0, lower bound: -0.5422272, upper bound: 0.5827478
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.42
Output dim: 0, lower bound: -0.5639934, upper bound: 0.5442548
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.42
Output dim: 0, lower bound: -0.5441623, upper bound: 0.5828222
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.42
Output dim: 0, lower bound: -0.5828222, upper bound: 0.5441623
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.42
Output dim: 0, lower bound: -0.5442548, upper bound: 0.5639934
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.42
Output dim: 0, lower bound: -0.5827478, upper bound: 0.5422272
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.42
Output dim: 0, lower bound: -0.5555948, upper bound: 0.5732267

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5439457, upper bound: 0.5553310
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5731152, upper bound: 0.5552399
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5411046, upper bound: 0.5824656
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5419741, upper bound: 0.5691682
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5395691, upper bound: 0.5826954
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5439453, upper bound: 0.5719572
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5719572, upper bound: 0.5439453
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5826954, upper bound: 0.5395691
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5691682, upper bound: 0.5419741
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5824656, upper bound: 0.5411046
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5552399, upper bound: 0.5731152
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5553310, upper bound: 0.5439457
time: 0.36 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.69 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.69
Output dim: 0, lower bound: -0.5439457, upper bound: 0.5553310
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.69
Output dim: 0, lower bound: -0.5731152, upper bound: 0.5552399
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.69
Output dim: 0, lower bound: -0.5411046, upper bound: 0.5824656
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.69
Output dim: 0, lower bound: -0.5419741, upper bound: 0.5691682
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.69
Output dim: 0, lower bound: -0.5395691, upper bound: 0.5826954
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.69
Output dim: 0, lower bound: -0.5439453, upper bound: 0.5719572
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.69
Output dim: 0, lower bound: -0.5719572, upper bound: 0.5439453
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.69
Output dim: 0, lower bound: -0.5826954, upper bound: 0.5395691
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.69
Output dim: 0, lower bound: -0.5691682, upper bound: 0.5419741
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.69
Output dim: 0, lower bound: -0.5824656, upper bound: 0.5411046
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.69
Output dim: 0, lower bound: -0.5552399, upper bound: 0.5731152
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.69
Output dim: 0, lower bound: -0.5553310, upper bound: 0.5439457

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5650934, upper bound: 0.5224687
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5731104, upper bound: 0.5551714
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5410553, upper bound: 0.5737050
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5340679, upper bound: 0.5824606
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5410553, upper bound: 0.5224687
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5401907, upper bound: 0.5688342
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5395141, upper bound: 0.5780496
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5308092, upper bound: 0.5826907
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5438800, upper bound: 0.5272497
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5438159, upper bound: 0.5715214
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5715214, upper bound: 0.5438159
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5272497, upper bound: 0.5438800
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5826907, upper bound: 0.5308092
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5780496, upper bound: 0.5395141
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5688342, upper bound: 0.5401907
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5224687, upper bound: 0.5419270
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5824606, upper bound: 0.5340679
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5737050, upper bound: 0.5410553
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5551714, upper bound: 0.5731104
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5224687, upper bound: 0.5650934
time: 0.35 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.34 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.34
Output dim: 0, lower bound: -0.5650934, upper bound: 0.5224687
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.34
Output dim: 0, lower bound: -0.5731104, upper bound: 0.5551714
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.34
Output dim: 0, lower bound: -0.5410553, upper bound: 0.5737050
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.34
Output dim: 0, lower bound: -0.5340679, upper bound: 0.5824606
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.34
Output dim: 0, lower bound: -0.5410553, upper bound: 0.5224687
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.34
Output dim: 0, lower bound: -0.5401907, upper bound: 0.5688342
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.34
Output dim: 0, lower bound: -0.5395141, upper bound: 0.5780496
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.34
Output dim: 0, lower bound: -0.5308092, upper bound: 0.5826907
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.34
Output dim: 0, lower bound: -0.5438800, upper bound: 0.5272497
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.34
Output dim: 0, lower bound: -0.5438159, upper bound: 0.5715214
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.34
Output dim: 0, lower bound: -0.5715214, upper bound: 0.5438159
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.34
Output dim: 0, lower bound: -0.5272497, upper bound: 0.5438800
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.34
Output dim: 0, lower bound: -0.5826907, upper bound: 0.5308092
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.34
Output dim: 0, lower bound: -0.5780496, upper bound: 0.5395141
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.34
Output dim: 0, lower bound: -0.5688342, upper bound: 0.5401907
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.34
Output dim: 0, lower bound: -0.5224687, upper bound: 0.5419270
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.34
Output dim: 0, lower bound: -0.5824606, upper bound: 0.5340679
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.34
Output dim: 0, lower bound: -0.5737050, upper bound: 0.5410553
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.34
Output dim: 0, lower bound: -0.5551714, upper bound: 0.5731104
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.34
Output dim: 0, lower bound: -0.5224687, upper bound: 0.5650934

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.24 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5634311, upper bound: 0.5365798
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5535169, upper bound: 0.5509407
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.24 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5339610, upper bound: 0.5590787
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203890, upper bound: 0.5677562
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.25 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5291521, upper bound: 0.5672077
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203789, upper bound: 0.5770248
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.23 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5337286, upper bound: 0.5507160
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5203878, upper bound: 0.5634436
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.25 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5331724, upper bound: 0.5633422
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203876, upper bound: 0.5712291
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.24 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5262855, upper bound: 0.5673497
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203719, upper bound: 0.5771027
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.26 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5373765, upper bound: 0.5542694
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5228855, upper bound: 0.5654198
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.27 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5654198, upper bound: 0.5228855
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5542694, upper bound: 0.5373765
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.25 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5771027, upper bound: 0.5203719
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5673497, upper bound: 0.5262855
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.24 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5712291, upper bound: 0.5203876
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5633422, upper bound: 0.5331724
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.27 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5634436, upper bound: 0.5203878
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5507160, upper bound: 0.5337286
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 23

Time for candidate selection: 1.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5770248, upper bound: 0.5203789
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5672077, upper bound: 0.5291521
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.27 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5677562, upper bound: 0.5203890
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5590787, upper bound: 0.5339610
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.31 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5509407, upper bound: 0.5535169
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5365798, upper bound: 0.5634311
time: 0.36 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.60 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.60
Output dim: 0, lower bound: -0.5634311, upper bound: 0.5365798
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.60
Output dim: 0, lower bound: -0.5535169, upper bound: 0.5509407
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.60
Output dim: 0, lower bound: -0.5339610, upper bound: 0.5590787
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -0.5203890, upper bound: 0.5677562
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -0.5291521, upper bound: 0.5672077
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -0.5203789, upper bound: 0.5770248
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.60
Output dim: 0, lower bound: -0.5337286, upper bound: 0.5507160
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.60
Output dim: 0, lower bound: -0.5203878, upper bound: 0.5634436
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.60
Output dim: 0, lower bound: -0.5331724, upper bound: 0.5633422
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -0.5203876, upper bound: 0.5712291
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -0.5262855, upper bound: 0.5673497
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -0.5203719, upper bound: 0.5771027
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.60
Output dim: 0, lower bound: -0.5373765, upper bound: 0.5542694
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -0.5228855, upper bound: 0.5654198
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -0.5654198, upper bound: 0.5228855
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.60
Output dim: 0, lower bound: -0.5542694, upper bound: 0.5373765
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -0.5771027, upper bound: 0.5203719
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -0.5673497, upper bound: 0.5262855
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -0.5712291, upper bound: 0.5203876
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.60
Output dim: 0, lower bound: -0.5633422, upper bound: 0.5331724
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.60
Output dim: 0, lower bound: -0.5634436, upper bound: 0.5203878
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.60
Output dim: 0, lower bound: -0.5507160, upper bound: 0.5337286
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -0.5770248, upper bound: 0.5203789
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -0.5672077, upper bound: 0.5291521
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -0.5677562, upper bound: 0.5203890
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.60
Output dim: 0, lower bound: -0.5590787, upper bound: 0.5339610
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.60
Output dim: 0, lower bound: -0.5509407, upper bound: 0.5535169
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.60
Output dim: 0, lower bound: -0.5365798, upper bound: 0.5634311

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5203640, upper bound: 0.5213139
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203276, upper bound: 0.5675479
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5291295, upper bound: 0.5224842
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5291006, upper bound: 0.5670215
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5203539, upper bound: 0.5256562
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203224, upper bound: 0.5768506
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5203625, upper bound: 0.5260726
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203268, upper bound: 0.5710205
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5262574, upper bound: 0.5264616
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5262423, upper bound: 0.5671637
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5203469, upper bound: 0.5296653
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203077, upper bound: 0.5769270
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5227134, upper bound: 0.5524678
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203187, upper bound: 0.5653871
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5653871, upper bound: 0.5204377
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5524678, upper bound: 0.5227134
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5168306, upper bound: 0.5134462
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5168306, upper bound: 0.5134462
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5149852, upper bound: 0.5134462
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5149852, upper bound: 0.5134462
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5165941, upper bound: 0.5134637
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5165941, upper bound: 0.5134630
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5768506, upper bound: 0.5203224
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5256562, upper bound: 0.5203539
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5670215, upper bound: 0.5291006
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5224842, upper bound: 0.5291295
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5143431, upper bound: 0.5134646
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5138966, upper bound: 0.5134638
time: 0.35 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.43 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.43
Output dim: 0, lower bound: -0.5203640, upper bound: 0.5213139
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 0, lower bound: -0.5203276, upper bound: 0.5675479
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.43
Output dim: 0, lower bound: -0.5291295, upper bound: 0.5224842
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 0, lower bound: -0.5291006, upper bound: 0.5670215
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.43
Output dim: 0, lower bound: -0.5203539, upper bound: 0.5256562
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 0, lower bound: -0.5203224, upper bound: 0.5768506
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.43
Output dim: 0, lower bound: -0.5203625, upper bound: 0.5260726
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 0, lower bound: -0.5203268, upper bound: 0.5710205
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.43
Output dim: 0, lower bound: -0.5262574, upper bound: 0.5264616
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 0, lower bound: -0.5262423, upper bound: 0.5671637
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.43
Output dim: 0, lower bound: -0.5203469, upper bound: 0.5296653
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 0, lower bound: -0.5203077, upper bound: 0.5769270
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.43
Output dim: 0, lower bound: -0.5227134, upper bound: 0.5524678
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 0, lower bound: -0.5203187, upper bound: 0.5653871
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 0, lower bound: -0.5653871, upper bound: 0.5204377
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.43
Output dim: 0, lower bound: -0.5524678, upper bound: 0.5227134
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.43
Output dim: 0, lower bound: -0.5168306, upper bound: 0.5134462
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.43
Output dim: 0, lower bound: -0.5168306, upper bound: 0.5134462
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.43
Output dim: 0, lower bound: -0.5149852, upper bound: 0.5134462
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.43
Output dim: 0, lower bound: -0.5149852, upper bound: 0.5134462
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.43
Output dim: 0, lower bound: -0.5165941, upper bound: 0.5134637
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.43
Output dim: 0, lower bound: -0.5165941, upper bound: 0.5134630
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 0, lower bound: -0.5768506, upper bound: 0.5203224
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.43
Output dim: 0, lower bound: -0.5256562, upper bound: 0.5203539
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.43
Output dim: 0, lower bound: -0.5670215, upper bound: 0.5291006
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.43
Output dim: 0, lower bound: -0.5224842, upper bound: 0.5291295
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.43
Output dim: 0, lower bound: -0.5143431, upper bound: 0.5134646
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.43
Output dim: 0, lower bound: -0.5138966, upper bound: 0.5134638

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5138510
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5143041
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133644, upper bound: 0.5134549
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133983, upper bound: 0.5134549
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5158407
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5158407
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5165533
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5165533
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5149469
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5149469
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5167898
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5167898
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5157573
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5157573
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5157573, upper bound: 0.5133628
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5146177, upper bound: 0.5133628
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5502034, upper bound: 0.5147006
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5522395, upper bound: 0.5146973
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5426386, upper bound: 0.5203462
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5443783, upper bound: 0.5168366
time: 0.34 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 2.45 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.45
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5138510
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.45
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5143041
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.45
Output dim: 0, lower bound: -0.5133644, upper bound: 0.5134549
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.45
Output dim: 0, lower bound: -0.5133983, upper bound: 0.5134549
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.45
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5158407
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.45
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5158407
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.45
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5165533
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.45
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5165533
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.45
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5149469
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.45
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5149469
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.45
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5167898
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.45
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5167898
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.45
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5157573
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.45
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5157573
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.45
Output dim: 0, lower bound: -0.5157573, upper bound: 0.5133628
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.45
Output dim: 0, lower bound: -0.5146177, upper bound: 0.5133628
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.45
Output dim: 0, lower bound: -0.5502034, upper bound: 0.5147006
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.45
Output dim: 0, lower bound: -0.5522395, upper bound: 0.5146973
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.45
Output dim: 0, lower bound: -0.5426386, upper bound: 0.5203462
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.45
Output dim: 0, lower bound: -0.5443783, upper bound: 0.5168366
Binary search (step 1): status=Status.VERIFIED, low=0.1406345, high=0.1818182, mid=0.1406345, abs_max=0.6789970397949219
rel_dist={0: [-0.5978996472348218, 0.597899647234821]}

## Binary search (step 2) starts
Candidate diff: 0.1612264


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5894953, upper bound: 0.5894953
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5894953, upper bound: 0.5894953
time: 0.36 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.86 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.86
Output dim: 0, lower bound: -0.5894953, upper bound: 0.5894953
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.86
Output dim: 0, lower bound: -0.5894953, upper bound: 0.5894953

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5868319, upper bound: 0.5832535
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5681019, upper bound: 0.5868319
time: 0.33 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5868319, upper bound: 0.5681019
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5832535, upper bound: 0.5868319
time: 0.35 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.31 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.31
Output dim: 0, lower bound: -0.5868319, upper bound: 0.5832535
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.31
Output dim: 0, lower bound: -0.5681019, upper bound: 0.5868319
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.31
Output dim: 0, lower bound: -0.5868319, upper bound: 0.5681019
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.31
Output dim: 0, lower bound: -0.5832535, upper bound: 0.5868319

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5732267, upper bound: 0.5555948
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5422272, upper bound: 0.5827478
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5639934, upper bound: 0.5442548
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5441623, upper bound: 0.5828222
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5828222, upper bound: 0.5441623
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5442548, upper bound: 0.5639934
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5827478, upper bound: 0.5422272
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5555948, upper bound: 0.5732267
time: 0.40 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.46 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 0, lower bound: -0.5732267, upper bound: 0.5555948
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 0, lower bound: -0.5422272, upper bound: 0.5827478
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.46
Output dim: 0, lower bound: -0.5639934, upper bound: 0.5442548
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 0, lower bound: -0.5441623, upper bound: 0.5828222
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 0, lower bound: -0.5828222, upper bound: 0.5441623
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.46
Output dim: 0, lower bound: -0.5442548, upper bound: 0.5639934
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 0, lower bound: -0.5827478, upper bound: 0.5422272
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 0, lower bound: -0.5555948, upper bound: 0.5732267

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5439457, upper bound: 0.5553310
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5731152, upper bound: 0.5552399
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5411046, upper bound: 0.5824656
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5419741, upper bound: 0.5691682
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5395691, upper bound: 0.5826954
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5439453, upper bound: 0.5719572
time: 0.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5719572, upper bound: 0.5439453
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5826954, upper bound: 0.5395691
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5691682, upper bound: 0.5419741
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5824656, upper bound: 0.5411046
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5552399, upper bound: 0.5731152
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5553310, upper bound: 0.5439457
time: 0.34 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.65 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 0, lower bound: -0.5439457, upper bound: 0.5553310
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 0, lower bound: -0.5731152, upper bound: 0.5552399
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 0, lower bound: -0.5411046, upper bound: 0.5824656
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 0, lower bound: -0.5419741, upper bound: 0.5691682
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 0, lower bound: -0.5395691, upper bound: 0.5826954
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 0, lower bound: -0.5439453, upper bound: 0.5719572
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 0, lower bound: -0.5719572, upper bound: 0.5439453
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 0, lower bound: -0.5826954, upper bound: 0.5395691
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 0, lower bound: -0.5691682, upper bound: 0.5419741
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 0, lower bound: -0.5824656, upper bound: 0.5411046
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 0, lower bound: -0.5552399, upper bound: 0.5731152
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 0, lower bound: -0.5553310, upper bound: 0.5439457

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5650934, upper bound: 0.5224687
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5731104, upper bound: 0.5551714
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5410553, upper bound: 0.5737050
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5340679, upper bound: 0.5824606
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5419270, upper bound: 0.5224687
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5401907, upper bound: 0.5688342
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5395141, upper bound: 0.5780496
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5308092, upper bound: 0.5826907
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5438800, upper bound: 0.5272497
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5438159, upper bound: 0.5715214
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5715214, upper bound: 0.5438159
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5272497, upper bound: 0.5438800
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5826907, upper bound: 0.5308092
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5780496, upper bound: 0.5395141
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5688342, upper bound: 0.5401907
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5224687, upper bound: 0.5419270
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5824606, upper bound: 0.5340679
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5737050, upper bound: 0.5410553
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5551714, upper bound: 0.5731104
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5224687, upper bound: 0.5650934
time: 0.33 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.41 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.5650934, upper bound: 0.5224687
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.5731104, upper bound: 0.5551714
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.5410553, upper bound: 0.5737050
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.5340679, upper bound: 0.5824606
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.5419270, upper bound: 0.5224687
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.5401907, upper bound: 0.5688342
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.5395141, upper bound: 0.5780496
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.5308092, upper bound: 0.5826907
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.5438800, upper bound: 0.5272497
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.5438159, upper bound: 0.5715214
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.5715214, upper bound: 0.5438159
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.5272497, upper bound: 0.5438800
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.5826907, upper bound: 0.5308092
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.5780496, upper bound: 0.5395141
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.5688342, upper bound: 0.5401907
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.5224687, upper bound: 0.5419270
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.5824606, upper bound: 0.5340679
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.5737050, upper bound: 0.5410553
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.5551714, upper bound: 0.5731104
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.5224687, upper bound: 0.5650934

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.24 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5634311, upper bound: 0.5365798
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5535169, upper bound: 0.5509407
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.26 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5339610, upper bound: 0.5590787
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203890, upper bound: 0.5677562
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.26 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5291521, upper bound: 0.5672077
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203789, upper bound: 0.5770248
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.28 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5337286, upper bound: 0.5507160
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5203878, upper bound: 0.5634436
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.26 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5331724, upper bound: 0.5633422
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203876, upper bound: 0.5712291
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.24 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5262855, upper bound: 0.5673497
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203719, upper bound: 0.5771027
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.28 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5373765, upper bound: 0.5542694
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5228855, upper bound: 0.5654198
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.27 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5654198, upper bound: 0.5228855
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5542694, upper bound: 0.5373765
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.25 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5771027, upper bound: 0.5203719
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5673497, upper bound: 0.5262855
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.28 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5712291, upper bound: 0.5203876
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5633422, upper bound: 0.5331724
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 23

Time for candidate selection: 1.28 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5634436, upper bound: 0.5203878
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5507160, upper bound: 0.5337286
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 23

Time for candidate selection: 1.27 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5770248, upper bound: 0.5203789
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5672077, upper bound: 0.5291521
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.31 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5677562, upper bound: 0.5203890
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5590787, upper bound: 0.5339610
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.32 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5509407, upper bound: 0.5535169
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5365798, upper bound: 0.5634311
time: 0.38 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.69 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.69
Output dim: 0, lower bound: -0.5634311, upper bound: 0.5365798
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.69
Output dim: 0, lower bound: -0.5535169, upper bound: 0.5509407
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.69
Output dim: 0, lower bound: -0.5339610, upper bound: 0.5590787
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 0, lower bound: -0.5203890, upper bound: 0.5677562
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 0, lower bound: -0.5291521, upper bound: 0.5672077
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 0, lower bound: -0.5203789, upper bound: 0.5770248
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.69
Output dim: 0, lower bound: -0.5337286, upper bound: 0.5507160
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.69
Output dim: 0, lower bound: -0.5203878, upper bound: 0.5634436
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.69
Output dim: 0, lower bound: -0.5331724, upper bound: 0.5633422
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 0, lower bound: -0.5203876, upper bound: 0.5712291
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 0, lower bound: -0.5262855, upper bound: 0.5673497
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 0, lower bound: -0.5203719, upper bound: 0.5771027
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.69
Output dim: 0, lower bound: -0.5373765, upper bound: 0.5542694
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 0, lower bound: -0.5228855, upper bound: 0.5654198
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 0, lower bound: -0.5654198, upper bound: 0.5228855
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.69
Output dim: 0, lower bound: -0.5542694, upper bound: 0.5373765
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 0, lower bound: -0.5771027, upper bound: 0.5203719
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 0, lower bound: -0.5673497, upper bound: 0.5262855
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 0, lower bound: -0.5712291, upper bound: 0.5203876
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.69
Output dim: 0, lower bound: -0.5633422, upper bound: 0.5331724
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.69
Output dim: 0, lower bound: -0.5634436, upper bound: 0.5203878
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.69
Output dim: 0, lower bound: -0.5507160, upper bound: 0.5337286
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 0, lower bound: -0.5770248, upper bound: 0.5203789
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 0, lower bound: -0.5672077, upper bound: 0.5291521
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.69
Output dim: 0, lower bound: -0.5677562, upper bound: 0.5203890
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.69
Output dim: 0, lower bound: -0.5590787, upper bound: 0.5339610
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.69
Output dim: 0, lower bound: -0.5509407, upper bound: 0.5535169
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.69
Output dim: 0, lower bound: -0.5365798, upper bound: 0.5634311

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5203640, upper bound: 0.5213139
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203276, upper bound: 0.5675479
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5291295, upper bound: 0.5224842
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5291006, upper bound: 0.5670215
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5203224, upper bound: 0.5256562
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203077, upper bound: 0.5768506
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5203625, upper bound: 0.5260726
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203268, upper bound: 0.5710205
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5262574, upper bound: 0.5264616
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5262423, upper bound: 0.5671637
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5203469, upper bound: 0.5296653
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203187, upper bound: 0.5769270
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5227134, upper bound: 0.5524678
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5204377, upper bound: 0.5653871
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5653871, upper bound: 0.5204377
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5524678, upper bound: 0.5227134
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5168306, upper bound: 0.5134462
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5168306, upper bound: 0.5134462
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5149852, upper bound: 0.5134462
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5146549, upper bound: 0.5134462
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5165941, upper bound: 0.5134637
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5165941, upper bound: 0.5134630
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5768506, upper bound: 0.5203224
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5256562, upper bound: 0.5203539
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5670215, upper bound: 0.5291006
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5224842, upper bound: 0.5291295
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5143431, upper bound: 0.5134646
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5138966, upper bound: 0.5134638
time: 0.36 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.45 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.45
Output dim: 0, lower bound: -0.5203640, upper bound: 0.5213139
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.45
Output dim: 0, lower bound: -0.5203276, upper bound: 0.5675479
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.45
Output dim: 0, lower bound: -0.5291295, upper bound: 0.5224842
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.45
Output dim: 0, lower bound: -0.5291006, upper bound: 0.5670215
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.45
Output dim: 0, lower bound: -0.5203224, upper bound: 0.5256562
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.45
Output dim: 0, lower bound: -0.5203077, upper bound: 0.5768506
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.45
Output dim: 0, lower bound: -0.5203625, upper bound: 0.5260726
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.45
Output dim: 0, lower bound: -0.5203268, upper bound: 0.5710205
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.45
Output dim: 0, lower bound: -0.5262574, upper bound: 0.5264616
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.45
Output dim: 0, lower bound: -0.5262423, upper bound: 0.5671637
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.45
Output dim: 0, lower bound: -0.5203469, upper bound: 0.5296653
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.45
Output dim: 0, lower bound: -0.5203187, upper bound: 0.5769270
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.45
Output dim: 0, lower bound: -0.5227134, upper bound: 0.5524678
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.45
Output dim: 0, lower bound: -0.5204377, upper bound: 0.5653871
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.45
Output dim: 0, lower bound: -0.5653871, upper bound: 0.5204377
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.45
Output dim: 0, lower bound: -0.5524678, upper bound: 0.5227134
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.45
Output dim: 0, lower bound: -0.5168306, upper bound: 0.5134462
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.45
Output dim: 0, lower bound: -0.5168306, upper bound: 0.5134462
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.45
Output dim: 0, lower bound: -0.5149852, upper bound: 0.5134462
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.45
Output dim: 0, lower bound: -0.5146549, upper bound: 0.5134462
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.45
Output dim: 0, lower bound: -0.5165941, upper bound: 0.5134637
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.45
Output dim: 0, lower bound: -0.5165941, upper bound: 0.5134630
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.45
Output dim: 0, lower bound: -0.5768506, upper bound: 0.5203224
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.45
Output dim: 0, lower bound: -0.5256562, upper bound: 0.5203539
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.45
Output dim: 0, lower bound: -0.5670215, upper bound: 0.5291006
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.45
Output dim: 0, lower bound: -0.5224842, upper bound: 0.5291295
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.45
Output dim: 0, lower bound: -0.5143431, upper bound: 0.5134646
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.45
Output dim: 0, lower bound: -0.5138966, upper bound: 0.5134638

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5138510
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5143041
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133644, upper bound: 0.5134549
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133983, upper bound: 0.5134549
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5158407
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5158407
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5165533
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5165533
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5149469
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5149469
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5167898
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5167898
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5157573
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5157573
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5157573, upper bound: 0.5133628
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5157573, upper bound: 0.5133628
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5502034, upper bound: 0.5147006
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5522395, upper bound: 0.5146973
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5426386, upper bound: 0.5203462
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5443783, upper bound: 0.5168366
time: 0.33 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 2.51 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.51
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5138510
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.51
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5143041
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.51
Output dim: 0, lower bound: -0.5133644, upper bound: 0.5134549
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.51
Output dim: 0, lower bound: -0.5133983, upper bound: 0.5134549
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.51
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5158407
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.51
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5158407
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.51
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5165533
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.51
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5165533
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.51
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5149469
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.51
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5149469
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.51
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5167898
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.51
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5167898
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.51
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5157573
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.51
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5157573
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.51
Output dim: 0, lower bound: -0.5157573, upper bound: 0.5133628
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.51
Output dim: 0, lower bound: -0.5157573, upper bound: 0.5133628
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.51
Output dim: 0, lower bound: -0.5502034, upper bound: 0.5147006
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.51
Output dim: 0, lower bound: -0.5522395, upper bound: 0.5146973
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.51
Output dim: 0, lower bound: -0.5426386, upper bound: 0.5203462
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.51
Output dim: 0, lower bound: -0.5443783, upper bound: 0.5168366
Binary search (step 2): status=Status.VERIFIED, low=0.1612264, high=0.1818182, mid=0.1612264, abs_max=0.6789970397949219
rel_dist={0: [-0.5989763754511993, 0.5989763754511985]}

## Binary search (step 3) starts
Candidate diff: 0.1715223


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5894953, upper bound: 0.5894953
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5894953, upper bound: 0.5894953
time: 0.35 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.83 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.83
Output dim: 0, lower bound: -0.5894953, upper bound: 0.5894953
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.83
Output dim: 0, lower bound: -0.5894953, upper bound: 0.5894953

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5868319, upper bound: 0.5832535
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5681019, upper bound: 0.5868319
time: 0.32 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5868319, upper bound: 0.5681019
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5832535, upper bound: 0.5868319
time: 0.34 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.28 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.28
Output dim: 0, lower bound: -0.5868319, upper bound: 0.5832535
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.28
Output dim: 0, lower bound: -0.5681019, upper bound: 0.5868319
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.28
Output dim: 0, lower bound: -0.5868319, upper bound: 0.5681019
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.28
Output dim: 0, lower bound: -0.5832535, upper bound: 0.5868319

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5732267, upper bound: 0.5555948
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5422272, upper bound: 0.5827478
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5639934, upper bound: 0.5442548
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5441623, upper bound: 0.5828222
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5828222, upper bound: 0.5441623
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5442548, upper bound: 0.5639934
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5827478, upper bound: 0.5422272
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5555948, upper bound: 0.5732267
time: 0.39 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.43 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.43
Output dim: 0, lower bound: -0.5732267, upper bound: 0.5555948
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.43
Output dim: 0, lower bound: -0.5422272, upper bound: 0.5827478
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.43
Output dim: 0, lower bound: -0.5639934, upper bound: 0.5442548
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.43
Output dim: 0, lower bound: -0.5441623, upper bound: 0.5828222
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.43
Output dim: 0, lower bound: -0.5828222, upper bound: 0.5441623
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.43
Output dim: 0, lower bound: -0.5442548, upper bound: 0.5639934
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.43
Output dim: 0, lower bound: -0.5827478, upper bound: 0.5422272
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.43
Output dim: 0, lower bound: -0.5555948, upper bound: 0.5732267

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5439457, upper bound: 0.5553310
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5731152, upper bound: 0.5552399
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5411046, upper bound: 0.5824656
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5419741, upper bound: 0.5691682
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5395691, upper bound: 0.5826954
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5439453, upper bound: 0.5719572
time: 0.43 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5719572, upper bound: 0.5439453
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5826954, upper bound: 0.5395691
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5691682, upper bound: 0.5419741
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5824656, upper bound: 0.5411046
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5552399, upper bound: 0.5731152
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5553310, upper bound: 0.5439457
time: 0.35 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.69 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.69
Output dim: 0, lower bound: -0.5439457, upper bound: 0.5553310
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.69
Output dim: 0, lower bound: -0.5731152, upper bound: 0.5552399
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.69
Output dim: 0, lower bound: -0.5411046, upper bound: 0.5824656
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.69
Output dim: 0, lower bound: -0.5419741, upper bound: 0.5691682
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.69
Output dim: 0, lower bound: -0.5395691, upper bound: 0.5826954
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.69
Output dim: 0, lower bound: -0.5439453, upper bound: 0.5719572
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.69
Output dim: 0, lower bound: -0.5719572, upper bound: 0.5439453
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.69
Output dim: 0, lower bound: -0.5826954, upper bound: 0.5395691
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.69
Output dim: 0, lower bound: -0.5691682, upper bound: 0.5419741
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.69
Output dim: 0, lower bound: -0.5824656, upper bound: 0.5411046
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.69
Output dim: 0, lower bound: -0.5552399, upper bound: 0.5731152
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.69
Output dim: 0, lower bound: -0.5553310, upper bound: 0.5439457

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5650934, upper bound: 0.5224687
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5731104, upper bound: 0.5551714
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5410553, upper bound: 0.5737050
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5340679, upper bound: 0.5824606
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5340679, upper bound: 0.5224687
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5401907, upper bound: 0.5688342
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5308092, upper bound: 0.5780496
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5308092, upper bound: 0.5826907
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5438800, upper bound: 0.5272497
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5438159, upper bound: 0.5715214
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5715214, upper bound: 0.5438159
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5224687, upper bound: 0.5438800
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5826907, upper bound: 0.5308092
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5780496, upper bound: 0.5395141
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5688342, upper bound: 0.5401907
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5224687, upper bound: 0.5419270
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5824606, upper bound: 0.5340679
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5737050, upper bound: 0.5410553
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5551714, upper bound: 0.5731104
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5224687, upper bound: 0.5650934
time: 0.34 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.41 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.5650934, upper bound: 0.5224687
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.5731104, upper bound: 0.5551714
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.5410553, upper bound: 0.5737050
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.5340679, upper bound: 0.5824606
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.5340679, upper bound: 0.5224687
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.5401907, upper bound: 0.5688342
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.5308092, upper bound: 0.5780496
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.5308092, upper bound: 0.5826907
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.5438800, upper bound: 0.5272497
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.5438159, upper bound: 0.5715214
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.5715214, upper bound: 0.5438159
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.5224687, upper bound: 0.5438800
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.5826907, upper bound: 0.5308092
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.5780496, upper bound: 0.5395141
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.5688342, upper bound: 0.5401907
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.5224687, upper bound: 0.5419270
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.5824606, upper bound: 0.5340679
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.5737050, upper bound: 0.5410553
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.5551714, upper bound: 0.5731104
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.5224687, upper bound: 0.5650934

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.28 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5634311, upper bound: 0.5365798
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5203359, upper bound: 0.5509407
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.25 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5339610, upper bound: 0.5590787
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203890, upper bound: 0.5677562
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.26 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5291521, upper bound: 0.5672077
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203789, upper bound: 0.5770248
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5337286, upper bound: 0.5507160
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5203878, upper bound: 0.5634436
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5331724, upper bound: 0.5633422
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203876, upper bound: 0.5712291
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.28 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5262855, upper bound: 0.5673497
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203719, upper bound: 0.5771027
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5373765, upper bound: 0.5542694
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5228855, upper bound: 0.5654198
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.31 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5654198, upper bound: 0.5228855
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5542694, upper bound: 0.5373765
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.28 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5771027, upper bound: 0.5203719
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5673497, upper bound: 0.5262855
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 5

Time for candidate selection: 1.32 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5712291, upper bound: 0.5203876
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5633422, upper bound: 0.5331724
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 23

Time for candidate selection: 1.31 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5634436, upper bound: 0.5203878
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5507160, upper bound: 0.5337286
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 23

Time for candidate selection: 1.32 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5770248, upper bound: 0.5203789
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5672077, upper bound: 0.5291521
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.28 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5677562, upper bound: 0.5203890
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5590787, upper bound: 0.5339610
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.32 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5509407, upper bound: 0.5535169
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5365798, upper bound: 0.5634311
time: 0.37 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.65 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.65
Output dim: 0, lower bound: -0.5634311, upper bound: 0.5365798
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.65
Output dim: 0, lower bound: -0.5203359, upper bound: 0.5509407
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.65
Output dim: 0, lower bound: -0.5339610, upper bound: 0.5590787
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -0.5203890, upper bound: 0.5677562
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -0.5291521, upper bound: 0.5672077
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -0.5203789, upper bound: 0.5770248
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.65
Output dim: 0, lower bound: -0.5337286, upper bound: 0.5507160
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.65
Output dim: 0, lower bound: -0.5203878, upper bound: 0.5634436
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.65
Output dim: 0, lower bound: -0.5331724, upper bound: 0.5633422
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -0.5203876, upper bound: 0.5712291
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -0.5262855, upper bound: 0.5673497
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -0.5203719, upper bound: 0.5771027
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.65
Output dim: 0, lower bound: -0.5373765, upper bound: 0.5542694
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -0.5228855, upper bound: 0.5654198
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -0.5654198, upper bound: 0.5228855
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.65
Output dim: 0, lower bound: -0.5542694, upper bound: 0.5373765
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -0.5771027, upper bound: 0.5203719
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -0.5673497, upper bound: 0.5262855
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -0.5712291, upper bound: 0.5203876
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.65
Output dim: 0, lower bound: -0.5633422, upper bound: 0.5331724
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.65
Output dim: 0, lower bound: -0.5634436, upper bound: 0.5203878
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.65
Output dim: 0, lower bound: -0.5507160, upper bound: 0.5337286
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -0.5770248, upper bound: 0.5203789
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -0.5672077, upper bound: 0.5291521
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -0.5677562, upper bound: 0.5203890
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.65
Output dim: 0, lower bound: -0.5590787, upper bound: 0.5339610
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.65
Output dim: 0, lower bound: -0.5509407, upper bound: 0.5535169
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.65
Output dim: 0, lower bound: -0.5365798, upper bound: 0.5634311

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5203640, upper bound: 0.5213139
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203276, upper bound: 0.5675479
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5291295, upper bound: 0.5224842
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5291006, upper bound: 0.5670215
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5203539, upper bound: 0.5256562
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203224, upper bound: 0.5768506
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5203625, upper bound: 0.5260726
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203268, upper bound: 0.5710205
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5262574, upper bound: 0.5264616
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5262423, upper bound: 0.5671637
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5203469, upper bound: 0.5296653
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203187, upper bound: 0.5769270
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5227134, upper bound: 0.5524678
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5204377, upper bound: 0.5653871
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5653871, upper bound: 0.5204377
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5524678, upper bound: 0.5227134
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5168306, upper bound: 0.5134462
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5168306, upper bound: 0.5134462
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5149852, upper bound: 0.5134462
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5149852, upper bound: 0.5134462
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5165941, upper bound: 0.5134637
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5165941, upper bound: 0.5134630
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5768506, upper bound: 0.5203224
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5256562, upper bound: 0.5203539
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5670215, upper bound: 0.5291006
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5224842, upper bound: 0.5291295
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5143431, upper bound: 0.5134646
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5138966, upper bound: 0.5134638
time: 0.37 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.48 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.5203640, upper bound: 0.5213139
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.5203276, upper bound: 0.5675479
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.5291295, upper bound: 0.5224842
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.5291006, upper bound: 0.5670215
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.5203539, upper bound: 0.5256562
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.5203224, upper bound: 0.5768506
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.5203625, upper bound: 0.5260726
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.5203268, upper bound: 0.5710205
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.5262574, upper bound: 0.5264616
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.5262423, upper bound: 0.5671637
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.5203469, upper bound: 0.5296653
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.5203187, upper bound: 0.5769270
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.5227134, upper bound: 0.5524678
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.5204377, upper bound: 0.5653871
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.5653871, upper bound: 0.5204377
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.5524678, upper bound: 0.5227134
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.5168306, upper bound: 0.5134462
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.5168306, upper bound: 0.5134462
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.5149852, upper bound: 0.5134462
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.5149852, upper bound: 0.5134462
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.5165941, upper bound: 0.5134637
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.5165941, upper bound: 0.5134630
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.5768506, upper bound: 0.5203224
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.5256562, upper bound: 0.5203539
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.5670215, upper bound: 0.5291006
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.5224842, upper bound: 0.5291295
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.5143431, upper bound: 0.5134646
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.5138966, upper bound: 0.5134638

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5138510
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5143041
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133644, upper bound: 0.5134549
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133983, upper bound: 0.5134549
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5158407
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5158407
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5165533
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5165533
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5149469
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5149469
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5167898
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5167898
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5157573
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5157573
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5138087, upper bound: 0.5133628
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5157573, upper bound: 0.5133628
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5502034, upper bound: 0.5147006
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5522395, upper bound: 0.5146973
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5426386, upper bound: 0.5203462
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5443783, upper bound: 0.5168366
time: 0.35 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 2.48 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5138510
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5143041
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 0, lower bound: -0.5133644, upper bound: 0.5134549
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 0, lower bound: -0.5133983, upper bound: 0.5134549
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5158407
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5158407
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5165533
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5165533
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5149469
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5149469
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5167898
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5167898
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5157573
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5157573
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 0, lower bound: -0.5138087, upper bound: 0.5133628
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 0, lower bound: -0.5157573, upper bound: 0.5133628
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 0, lower bound: -0.5502034, upper bound: 0.5147006
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 0, lower bound: -0.5522395, upper bound: 0.5146973
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 0, lower bound: -0.5426386, upper bound: 0.5203462
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 0, lower bound: -0.5443783, upper bound: 0.5168366
Binary search (step 3): status=Status.VERIFIED, low=0.1715223, high=0.1818182, mid=0.1715223, abs_max=0.6789970397949219
rel_dist={0: [-0.5994456166514532, 0.5994456166514528]}

## Binary search (step 4) starts
Candidate diff: 0.1766702


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5894953, upper bound: 0.5894953
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5894953, upper bound: 0.5894953
time: 0.35 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.83 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.83
Output dim: 0, lower bound: -0.5894953, upper bound: 0.5894953
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.83
Output dim: 0, lower bound: -0.5894953, upper bound: 0.5894953

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5868319, upper bound: 0.5832535
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5681019, upper bound: 0.5868319
time: 0.31 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5868319, upper bound: 0.5681019
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5832535, upper bound: 0.5868319
time: 0.36 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.30 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.30
Output dim: 0, lower bound: -0.5868319, upper bound: 0.5832535
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.30
Output dim: 0, lower bound: -0.5681019, upper bound: 0.5868319
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.30
Output dim: 0, lower bound: -0.5868319, upper bound: 0.5681019
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.30
Output dim: 0, lower bound: -0.5832535, upper bound: 0.5868319

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5732267, upper bound: 0.5555948
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5422272, upper bound: 0.5827478
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5639934, upper bound: 0.5442548
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5441623, upper bound: 0.5828222
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5828222, upper bound: 0.5441623
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5442548, upper bound: 0.5639934
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5827478, upper bound: 0.5422272
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5555948, upper bound: 0.5732267
time: 0.44 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.47 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.47
Output dim: 0, lower bound: -0.5732267, upper bound: 0.5555948
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.47
Output dim: 0, lower bound: -0.5422272, upper bound: 0.5827478
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.47
Output dim: 0, lower bound: -0.5639934, upper bound: 0.5442548
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.47
Output dim: 0, lower bound: -0.5441623, upper bound: 0.5828222
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.47
Output dim: 0, lower bound: -0.5828222, upper bound: 0.5441623
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.47
Output dim: 0, lower bound: -0.5442548, upper bound: 0.5639934
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.47
Output dim: 0, lower bound: -0.5827478, upper bound: 0.5422272
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.47
Output dim: 0, lower bound: -0.5555948, upper bound: 0.5732267

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5439457, upper bound: 0.5553310
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5731152, upper bound: 0.5552399
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5411046, upper bound: 0.5824656
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5419741, upper bound: 0.5691682
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5395691, upper bound: 0.5826954
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5439453, upper bound: 0.5719572
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5719572, upper bound: 0.5439453
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5826954, upper bound: 0.5395691
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5691682, upper bound: 0.5419741
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5824656, upper bound: 0.5411046
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5552399, upper bound: 0.5731152
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5553310, upper bound: 0.5439457
time: 0.34 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.65 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 0, lower bound: -0.5439457, upper bound: 0.5553310
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 0, lower bound: -0.5731152, upper bound: 0.5552399
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 0, lower bound: -0.5411046, upper bound: 0.5824656
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 0, lower bound: -0.5419741, upper bound: 0.5691682
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 0, lower bound: -0.5395691, upper bound: 0.5826954
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 0, lower bound: -0.5439453, upper bound: 0.5719572
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 0, lower bound: -0.5719572, upper bound: 0.5439453
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 0, lower bound: -0.5826954, upper bound: 0.5395691
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 0, lower bound: -0.5691682, upper bound: 0.5419741
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 0, lower bound: -0.5824656, upper bound: 0.5411046
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 0, lower bound: -0.5552399, upper bound: 0.5731152
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 0, lower bound: -0.5553310, upper bound: 0.5439457

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5650934, upper bound: 0.5224687
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5731104, upper bound: 0.5551714
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5410553, upper bound: 0.5737050
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5340679, upper bound: 0.5824606
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5419270, upper bound: 0.5224687
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5401907, upper bound: 0.5688342
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5395141, upper bound: 0.5780496
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5308092, upper bound: 0.5826907
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5438800, upper bound: 0.5272497
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5438159, upper bound: 0.5715214
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5715214, upper bound: 0.5438159
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5272497, upper bound: 0.5438800
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5826907, upper bound: 0.5308092
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5780496, upper bound: 0.5395141
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5688342, upper bound: 0.5401907
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5224687, upper bound: 0.5419270
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5824606, upper bound: 0.5340679
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5737050, upper bound: 0.5410553
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5551714, upper bound: 0.5731104
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5224687, upper bound: 0.5650934
time: 0.34 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.37 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.37
Output dim: 0, lower bound: -0.5650934, upper bound: 0.5224687
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 0, lower bound: -0.5731104, upper bound: 0.5551714
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 0, lower bound: -0.5410553, upper bound: 0.5737050
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 0, lower bound: -0.5340679, upper bound: 0.5824606
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.37
Output dim: 0, lower bound: -0.5419270, upper bound: 0.5224687
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 0, lower bound: -0.5401907, upper bound: 0.5688342
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 0, lower bound: -0.5395141, upper bound: 0.5780496
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 0, lower bound: -0.5308092, upper bound: 0.5826907
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.37
Output dim: 0, lower bound: -0.5438800, upper bound: 0.5272497
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 0, lower bound: -0.5438159, upper bound: 0.5715214
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 0, lower bound: -0.5715214, upper bound: 0.5438159
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.37
Output dim: 0, lower bound: -0.5272497, upper bound: 0.5438800
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 0, lower bound: -0.5826907, upper bound: 0.5308092
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 0, lower bound: -0.5780496, upper bound: 0.5395141
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 0, lower bound: -0.5688342, upper bound: 0.5401907
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.37
Output dim: 0, lower bound: -0.5224687, upper bound: 0.5419270
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 0, lower bound: -0.5824606, upper bound: 0.5340679
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 0, lower bound: -0.5737050, upper bound: 0.5410553
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 0, lower bound: -0.5551714, upper bound: 0.5731104
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.37
Output dim: 0, lower bound: -0.5224687, upper bound: 0.5650934

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5634311, upper bound: 0.5365798
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5535169, upper bound: 0.5509407
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.26 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5203890, upper bound: 0.5590787
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203890, upper bound: 0.5677562
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.28 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5291521, upper bound: 0.5672077
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203789, upper bound: 0.5770248
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.31 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5337286, upper bound: 0.5507160
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5203878, upper bound: 0.5634436
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.27 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5331724, upper bound: 0.5633422
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203876, upper bound: 0.5712291
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.32 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5262855, upper bound: 0.5673497
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203719, upper bound: 0.5771027
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5373765, upper bound: 0.5542694
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5228855, upper bound: 0.5654198
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.28 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5654198, upper bound: 0.5228855
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5542694, upper bound: 0.5373765
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.28 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5771027, upper bound: 0.5203719
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5673497, upper bound: 0.5262855
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 5

Time for candidate selection: 1.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5712291, upper bound: 0.5203876
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5633422, upper bound: 0.5331724
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 23

Time for candidate selection: 1.31 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5634436, upper bound: 0.5203878
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5203517, upper bound: 0.5337286
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 23

Time for candidate selection: 1.28 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5770248, upper bound: 0.5203789
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5672077, upper bound: 0.5291521
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5677562, upper bound: 0.5203890
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5590787, upper bound: 0.5339610
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5509407, upper bound: 0.5535169
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5365798, upper bound: 0.5634311
time: 0.37 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.63 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.5634311, upper bound: 0.5365798
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.5535169, upper bound: 0.5509407
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.5203890, upper bound: 0.5590787
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.5203890, upper bound: 0.5677562
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.5291521, upper bound: 0.5672077
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.5203789, upper bound: 0.5770248
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.5337286, upper bound: 0.5507160
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.5203878, upper bound: 0.5634436
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.5331724, upper bound: 0.5633422
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.5203876, upper bound: 0.5712291
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.5262855, upper bound: 0.5673497
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.5203719, upper bound: 0.5771027
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.5373765, upper bound: 0.5542694
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.5228855, upper bound: 0.5654198
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.5654198, upper bound: 0.5228855
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.5542694, upper bound: 0.5373765
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.5771027, upper bound: 0.5203719
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.5673497, upper bound: 0.5262855
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.5712291, upper bound: 0.5203876
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.5633422, upper bound: 0.5331724
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.5634436, upper bound: 0.5203878
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.5203517, upper bound: 0.5337286
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.5770248, upper bound: 0.5203789
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.5672077, upper bound: 0.5291521
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.5677562, upper bound: 0.5203890
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.5590787, upper bound: 0.5339610
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.5509407, upper bound: 0.5535169
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.5365798, upper bound: 0.5634311

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5203640, upper bound: 0.5213139
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203276, upper bound: 0.5675479
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5291295, upper bound: 0.5224842
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5291006, upper bound: 0.5670215
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5203539, upper bound: 0.5256562
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203224, upper bound: 0.5768506
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5203625, upper bound: 0.5260726
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203268, upper bound: 0.5710205
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5262574, upper bound: 0.5264616
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5262423, upper bound: 0.5671637
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5203265, upper bound: 0.5296653
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203187, upper bound: 0.5769270
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5227134, upper bound: 0.5524678
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5204377, upper bound: 0.5653871
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5653871, upper bound: 0.5204377
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5524678, upper bound: 0.5227134
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5168306, upper bound: 0.5134462
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5168306, upper bound: 0.5134462
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5149852, upper bound: 0.5134462
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5149852, upper bound: 0.5134462
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5165941, upper bound: 0.5134637
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5165941, upper bound: 0.5134630
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5768506, upper bound: 0.5203224
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5256562, upper bound: 0.5203539
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5203077, upper bound: 0.5291006
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5224842, upper bound: 0.5291295
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5143431, upper bound: 0.5134646
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5138966, upper bound: 0.5134638
time: 0.34 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.39 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.39
Output dim: 0, lower bound: -0.5203640, upper bound: 0.5213139
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 0, lower bound: -0.5203276, upper bound: 0.5675479
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.39
Output dim: 0, lower bound: -0.5291295, upper bound: 0.5224842
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 0, lower bound: -0.5291006, upper bound: 0.5670215
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.39
Output dim: 0, lower bound: -0.5203539, upper bound: 0.5256562
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 0, lower bound: -0.5203224, upper bound: 0.5768506
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.39
Output dim: 0, lower bound: -0.5203625, upper bound: 0.5260726
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 0, lower bound: -0.5203268, upper bound: 0.5710205
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.39
Output dim: 0, lower bound: -0.5262574, upper bound: 0.5264616
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 0, lower bound: -0.5262423, upper bound: 0.5671637
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.39
Output dim: 0, lower bound: -0.5203265, upper bound: 0.5296653
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 0, lower bound: -0.5203187, upper bound: 0.5769270
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.39
Output dim: 0, lower bound: -0.5227134, upper bound: 0.5524678
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 0, lower bound: -0.5204377, upper bound: 0.5653871
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 0, lower bound: -0.5653871, upper bound: 0.5204377
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.39
Output dim: 0, lower bound: -0.5524678, upper bound: 0.5227134
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.39
Output dim: 0, lower bound: -0.5168306, upper bound: 0.5134462
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.39
Output dim: 0, lower bound: -0.5168306, upper bound: 0.5134462
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.39
Output dim: 0, lower bound: -0.5149852, upper bound: 0.5134462
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.39
Output dim: 0, lower bound: -0.5149852, upper bound: 0.5134462
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.39
Output dim: 0, lower bound: -0.5165941, upper bound: 0.5134637
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.39
Output dim: 0, lower bound: -0.5165941, upper bound: 0.5134630
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 0, lower bound: -0.5768506, upper bound: 0.5203224
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.39
Output dim: 0, lower bound: -0.5256562, upper bound: 0.5203539
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.39
Output dim: 0, lower bound: -0.5203077, upper bound: 0.5291006
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.39
Output dim: 0, lower bound: -0.5224842, upper bound: 0.5291295
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.39
Output dim: 0, lower bound: -0.5143431, upper bound: 0.5134646
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.39
Output dim: 0, lower bound: -0.5138966, upper bound: 0.5134638

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5138510
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5143041
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133644, upper bound: 0.5134549
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133983, upper bound: 0.5134549
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5158407
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5158407
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5165533
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5165533
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5149469
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5149469
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5167898
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5167898
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5157573
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5157573
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5157573, upper bound: 0.5133628
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5157573, upper bound: 0.5133628
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5502034, upper bound: 0.5147006
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5522395, upper bound: 0.5146973
time: 0.37 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 2.50 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.50
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5138510
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.50
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5143041
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.50
Output dim: 0, lower bound: -0.5133644, upper bound: 0.5134549
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.50
Output dim: 0, lower bound: -0.5133983, upper bound: 0.5134549
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.50
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5158407
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.50
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5158407
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.50
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5165533
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.50
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5165533
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.50
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5149469
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.50
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5149469
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.50
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5167898
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.50
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5167898
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.50
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5157573
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.50
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5157573
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.50
Output dim: 0, lower bound: -0.5157573, upper bound: 0.5133628
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.50
Output dim: 0, lower bound: -0.5157573, upper bound: 0.5133628
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.50
Output dim: 0, lower bound: -0.5502034, upper bound: 0.5147006
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.50
Output dim: 0, lower bound: -0.5522395, upper bound: 0.5146973
Binary search (step 4): status=Status.VERIFIED, low=0.1766702, high=0.1818182, mid=0.1766702, abs_max=0.6789970397949219
rel_dist={0: [-0.5996024140649461, 0.5996024140649459]}

## Binary search (step 5) starts
Candidate diff: 0.1792442


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5894953, upper bound: 0.5894953
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5894953, upper bound: 0.5894953
time: 0.33 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.82 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.82
Output dim: 0, lower bound: -0.5894953, upper bound: 0.5894953
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.82
Output dim: 0, lower bound: -0.5894953, upper bound: 0.5894953

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5868319, upper bound: 0.5832535
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5681019, upper bound: 0.5868319
time: 0.31 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5832535, upper bound: 0.5681019
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5832535, upper bound: 0.5868319
time: 0.35 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.35 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.35
Output dim: 0, lower bound: -0.5868319, upper bound: 0.5832535
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.35
Output dim: 0, lower bound: -0.5681019, upper bound: 0.5868319
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.35
Output dim: 0, lower bound: -0.5832535, upper bound: 0.5681019
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.35
Output dim: 0, lower bound: -0.5832535, upper bound: 0.5868319

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5732267, upper bound: 0.5555948
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5422272, upper bound: 0.5827478
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5639934, upper bound: 0.5442548
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5441623, upper bound: 0.5828222
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5828222, upper bound: 0.5441623
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5442548, upper bound: 0.5639934
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5827478, upper bound: 0.5422272
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5555948, upper bound: 0.5732267
time: 0.38 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.43 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.43
Output dim: 0, lower bound: -0.5732267, upper bound: 0.5555948
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.43
Output dim: 0, lower bound: -0.5422272, upper bound: 0.5827478
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.43
Output dim: 0, lower bound: -0.5639934, upper bound: 0.5442548
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.43
Output dim: 0, lower bound: -0.5441623, upper bound: 0.5828222
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.43
Output dim: 0, lower bound: -0.5828222, upper bound: 0.5441623
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.43
Output dim: 0, lower bound: -0.5442548, upper bound: 0.5639934
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.43
Output dim: 0, lower bound: -0.5827478, upper bound: 0.5422272
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.43
Output dim: 0, lower bound: -0.5555948, upper bound: 0.5732267

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5439457, upper bound: 0.5553310
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5225306, upper bound: 0.5552399
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5411046, upper bound: 0.5824656
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5419741, upper bound: 0.5691682
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5395691, upper bound: 0.5826954
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5439453, upper bound: 0.5719572
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5440061, upper bound: 0.5439453
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5826954, upper bound: 0.5395691
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5691682, upper bound: 0.5419741
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5824656, upper bound: 0.5411046
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5552399, upper bound: 0.5731152
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5553310, upper bound: 0.5439457
time: 0.34 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.65 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 0, lower bound: -0.5439457, upper bound: 0.5553310
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 0, lower bound: -0.5225306, upper bound: 0.5552399
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 0, lower bound: -0.5411046, upper bound: 0.5824656
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 0, lower bound: -0.5419741, upper bound: 0.5691682
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 0, lower bound: -0.5395691, upper bound: 0.5826954
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 0, lower bound: -0.5439453, upper bound: 0.5719572
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 0, lower bound: -0.5440061, upper bound: 0.5439453
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 0, lower bound: -0.5826954, upper bound: 0.5395691
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 0, lower bound: -0.5691682, upper bound: 0.5419741
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 0, lower bound: -0.5824656, upper bound: 0.5411046
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 0, lower bound: -0.5552399, upper bound: 0.5731152
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 0, lower bound: -0.5553310, upper bound: 0.5439457

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5410553, upper bound: 0.5737050
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5340679, upper bound: 0.5824606
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5419270, upper bound: 0.5224687
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5401907, upper bound: 0.5688342
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5395141, upper bound: 0.5780496
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5308092, upper bound: 0.5826907
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5438800, upper bound: 0.5272497
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5438159, upper bound: 0.5715214
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5826907, upper bound: 0.5308092
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5780496, upper bound: 0.5395141
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5688342, upper bound: 0.5401907
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5224687, upper bound: 0.5419270
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5824606, upper bound: 0.5340679
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5737050, upper bound: 0.5410553
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5551714, upper bound: 0.5731104
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5224687, upper bound: 0.5650934
time: 0.36 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.41 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.5410553, upper bound: 0.5737050
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.5340679, upper bound: 0.5824606
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.5419270, upper bound: 0.5224687
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.5401907, upper bound: 0.5688342
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.5395141, upper bound: 0.5780496
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.5308092, upper bound: 0.5826907
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.5438800, upper bound: 0.5272497
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.5438159, upper bound: 0.5715214
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.5826907, upper bound: 0.5308092
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.5780496, upper bound: 0.5395141
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.5688342, upper bound: 0.5401907
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.5224687, upper bound: 0.5419270
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.5824606, upper bound: 0.5340679
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.5737050, upper bound: 0.5410553
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.5551714, upper bound: 0.5731104
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.5224687, upper bound: 0.5650934

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.25 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5339610, upper bound: 0.5590787
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203890, upper bound: 0.5677562
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5291521, upper bound: 0.5672077
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203789, upper bound: 0.5770248
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5337286, upper bound: 0.5507160
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5203878, upper bound: 0.5634436
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.27 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5331724, upper bound: 0.5633422
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203876, upper bound: 0.5712291
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5262855, upper bound: 0.5673497
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203719, upper bound: 0.5771027
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.25 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5373765, upper bound: 0.5542694
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5228855, upper bound: 0.5654198
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.26 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5771027, upper bound: 0.5203719
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5673497, upper bound: 0.5262855
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 5

Time for candidate selection: 1.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5712291, upper bound: 0.5203876
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5633422, upper bound: 0.5331724
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 23

Time for candidate selection: 1.31 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5634436, upper bound: 0.5203878
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5507160, upper bound: 0.5337286
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 23

Time for candidate selection: 1.25 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5770248, upper bound: 0.5203789
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5672077, upper bound: 0.5291521
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 5

Time for candidate selection: 1.32 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5677562, upper bound: 0.5203890
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5590787, upper bound: 0.5339610
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5509407, upper bound: 0.5535169
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5365798, upper bound: 0.5634311
time: 0.35 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.54 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.54
Output dim: 0, lower bound: -0.5339610, upper bound: 0.5590787
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 0, lower bound: -0.5203890, upper bound: 0.5677562
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 0, lower bound: -0.5291521, upper bound: 0.5672077
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 0, lower bound: -0.5203789, upper bound: 0.5770248
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.54
Output dim: 0, lower bound: -0.5337286, upper bound: 0.5507160
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.54
Output dim: 0, lower bound: -0.5203878, upper bound: 0.5634436
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.54
Output dim: 0, lower bound: -0.5331724, upper bound: 0.5633422
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 0, lower bound: -0.5203876, upper bound: 0.5712291
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 0, lower bound: -0.5262855, upper bound: 0.5673497
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 0, lower bound: -0.5203719, upper bound: 0.5771027
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.54
Output dim: 0, lower bound: -0.5373765, upper bound: 0.5542694
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 0, lower bound: -0.5228855, upper bound: 0.5654198
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 0, lower bound: -0.5771027, upper bound: 0.5203719
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 0, lower bound: -0.5673497, upper bound: 0.5262855
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 0, lower bound: -0.5712291, upper bound: 0.5203876
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.54
Output dim: 0, lower bound: -0.5633422, upper bound: 0.5331724
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.54
Output dim: 0, lower bound: -0.5634436, upper bound: 0.5203878
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.54
Output dim: 0, lower bound: -0.5507160, upper bound: 0.5337286
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 0, lower bound: -0.5770248, upper bound: 0.5203789
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 0, lower bound: -0.5672077, upper bound: 0.5291521
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 0, lower bound: -0.5677562, upper bound: 0.5203890
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.54
Output dim: 0, lower bound: -0.5590787, upper bound: 0.5339610
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.54
Output dim: 0, lower bound: -0.5509407, upper bound: 0.5535169
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.54
Output dim: 0, lower bound: -0.5365798, upper bound: 0.5634311

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5203640, upper bound: 0.5213139
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203276, upper bound: 0.5675479
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5291295, upper bound: 0.5224842
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5291006, upper bound: 0.5670215
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5203539, upper bound: 0.5256562
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203224, upper bound: 0.5768506
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5203625, upper bound: 0.5260726
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203268, upper bound: 0.5710205
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5203469, upper bound: 0.5264616
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5262423, upper bound: 0.5671637
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5203469, upper bound: 0.5296653
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203187, upper bound: 0.5769270
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5227134, upper bound: 0.5524678
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5204377, upper bound: 0.5653871
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5168306, upper bound: 0.5134462
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5168306, upper bound: 0.5134462
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5149852, upper bound: 0.5134462
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5149852, upper bound: 0.5134462
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5165941, upper bound: 0.5134637
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5165941, upper bound: 0.5134630
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5768506, upper bound: 0.5203224
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5256562, upper bound: 0.5203539
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5670215, upper bound: 0.5291006
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5224842, upper bound: 0.5291295
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5143431, upper bound: 0.5134646
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5138966, upper bound: 0.5134638
time: 0.38 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.46 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.5203640, upper bound: 0.5213139
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.5203276, upper bound: 0.5675479
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.5291295, upper bound: 0.5224842
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.5291006, upper bound: 0.5670215
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.5203539, upper bound: 0.5256562
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.5203224, upper bound: 0.5768506
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.5203625, upper bound: 0.5260726
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.5203268, upper bound: 0.5710205
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.5203469, upper bound: 0.5264616
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.5262423, upper bound: 0.5671637
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.5203469, upper bound: 0.5296653
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.5203187, upper bound: 0.5769270
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.5227134, upper bound: 0.5524678
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.5204377, upper bound: 0.5653871
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.5168306, upper bound: 0.5134462
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.5168306, upper bound: 0.5134462
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.5149852, upper bound: 0.5134462
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.5149852, upper bound: 0.5134462
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.5165941, upper bound: 0.5134637
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.5165941, upper bound: 0.5134630
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.5768506, upper bound: 0.5203224
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.5256562, upper bound: 0.5203539
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.5670215, upper bound: 0.5291006
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.5224842, upper bound: 0.5291295
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.5143431, upper bound: 0.5134646
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.5138966, upper bound: 0.5134638

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5138510
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5143041
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133644, upper bound: 0.5134549
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133983, upper bound: 0.5134549
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5158407
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5158407
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5165533
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5165533
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5149469
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5149469
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5167898
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5167898
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5157573
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5157573
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5502034, upper bound: 0.5147006
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5522395, upper bound: 0.5146973
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5426386, upper bound: 0.5203462
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5443783, upper bound: 0.5168366
time: 0.35 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 2.52 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.52
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5138510
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.52
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5143041
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.52
Output dim: 0, lower bound: -0.5133644, upper bound: 0.5134549
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.52
Output dim: 0, lower bound: -0.5133983, upper bound: 0.5134549
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.52
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5158407
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.52
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5158407
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.52
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5165533
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.52
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5165533
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.52
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5149469
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.52
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5149469
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.52
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5167898
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.52
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5167898
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.52
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5157573
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.52
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5157573
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.52
Output dim: 0, lower bound: -0.5502034, upper bound: 0.5147006
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.52
Output dim: 0, lower bound: -0.5522395, upper bound: 0.5146973
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.52
Output dim: 0, lower bound: -0.5426386, upper bound: 0.5203462
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.52
Output dim: 0, lower bound: -0.5443783, upper bound: 0.5168366
Binary search (step 5): status=Status.VERIFIED, low=0.1792442, high=0.1818182, mid=0.1792442, abs_max=0.6789970397949219
rel_dist={0: [-0.5996291232441683, 0.5996291232441684]}

## Binary search (step 6) starts
Candidate diff: 0.1805312


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5894953, upper bound: 0.5894953
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5894953, upper bound: 0.5894953
time: 0.32 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.82 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.82
Output dim: 0, lower bound: -0.5894953, upper bound: 0.5894953
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.82
Output dim: 0, lower bound: -0.5894953, upper bound: 0.5894953

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5868319, upper bound: 0.5832535
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5681019, upper bound: 0.5868319
time: 0.30 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5868319, upper bound: 0.5681019
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5832535, upper bound: 0.5868319
time: 0.38 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.35 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.35
Output dim: 0, lower bound: -0.5868319, upper bound: 0.5832535
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.35
Output dim: 0, lower bound: -0.5681019, upper bound: 0.5868319
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.35
Output dim: 0, lower bound: -0.5868319, upper bound: 0.5681019
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.35
Output dim: 0, lower bound: -0.5832535, upper bound: 0.5868319

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5732267, upper bound: 0.5555948
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5422272, upper bound: 0.5827478
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5639934, upper bound: 0.5442548
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5441623, upper bound: 0.5828222
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5828222, upper bound: 0.5441623
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5442548, upper bound: 0.5639934
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5827478, upper bound: 0.5422272
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5555948, upper bound: 0.5732267
time: 0.41 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.43 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.43
Output dim: 0, lower bound: -0.5732267, upper bound: 0.5555948
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.43
Output dim: 0, lower bound: -0.5422272, upper bound: 0.5827478
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.43
Output dim: 0, lower bound: -0.5639934, upper bound: 0.5442548
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.43
Output dim: 0, lower bound: -0.5441623, upper bound: 0.5828222
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.43
Output dim: 0, lower bound: -0.5828222, upper bound: 0.5441623
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.43
Output dim: 0, lower bound: -0.5442548, upper bound: 0.5639934
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.43
Output dim: 0, lower bound: -0.5827478, upper bound: 0.5422272
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.43
Output dim: 0, lower bound: -0.5555948, upper bound: 0.5732267

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5439457, upper bound: 0.5553310
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5731152, upper bound: 0.5552399
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5411046, upper bound: 0.5824656
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5419741, upper bound: 0.5691682
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5395691, upper bound: 0.5826954
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5439453, upper bound: 0.5719572
time: 0.43 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5719572, upper bound: 0.5439453
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5826954, upper bound: 0.5395691
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5691682, upper bound: 0.5419741
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5824656, upper bound: 0.5411046
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5552399, upper bound: 0.5731152
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5553310, upper bound: 0.5439457
time: 0.36 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.80 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.80
Output dim: 0, lower bound: -0.5439457, upper bound: 0.5553310
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 0, lower bound: -0.5731152, upper bound: 0.5552399
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 0, lower bound: -0.5411046, upper bound: 0.5824656
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 0, lower bound: -0.5419741, upper bound: 0.5691682
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 0, lower bound: -0.5395691, upper bound: 0.5826954
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 0, lower bound: -0.5439453, upper bound: 0.5719572
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 0, lower bound: -0.5719572, upper bound: 0.5439453
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 0, lower bound: -0.5826954, upper bound: 0.5395691
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 0, lower bound: -0.5691682, upper bound: 0.5419741
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 0, lower bound: -0.5824656, upper bound: 0.5411046
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 0, lower bound: -0.5552399, upper bound: 0.5731152
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.80
Output dim: 0, lower bound: -0.5553310, upper bound: 0.5439457

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5650934, upper bound: 0.5224687
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5731104, upper bound: 0.5551714
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5410553, upper bound: 0.5737050
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5340679, upper bound: 0.5824606
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5419270, upper bound: 0.5224687
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5401907, upper bound: 0.5688342
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5308092, upper bound: 0.5780496
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5308092, upper bound: 0.5826907
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5438800, upper bound: 0.5272497
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5438159, upper bound: 0.5715214
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5715214, upper bound: 0.5438159
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5272497, upper bound: 0.5438800
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5826907, upper bound: 0.5308092
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5780496, upper bound: 0.5395141
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5688342, upper bound: 0.5401907
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5224687, upper bound: 0.5419270
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5824606, upper bound: 0.5340679
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5737050, upper bound: 0.5410553
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5551714, upper bound: 0.5731104
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5224687, upper bound: 0.5650934
time: 0.34 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.42 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.42
Output dim: 0, lower bound: -0.5650934, upper bound: 0.5224687
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -0.5731104, upper bound: 0.5551714
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -0.5410553, upper bound: 0.5737050
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -0.5340679, upper bound: 0.5824606
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.42
Output dim: 0, lower bound: -0.5419270, upper bound: 0.5224687
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -0.5401907, upper bound: 0.5688342
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -0.5308092, upper bound: 0.5780496
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -0.5308092, upper bound: 0.5826907
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.42
Output dim: 0, lower bound: -0.5438800, upper bound: 0.5272497
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -0.5438159, upper bound: 0.5715214
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -0.5715214, upper bound: 0.5438159
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.42
Output dim: 0, lower bound: -0.5272497, upper bound: 0.5438800
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -0.5826907, upper bound: 0.5308092
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -0.5780496, upper bound: 0.5395141
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -0.5688342, upper bound: 0.5401907
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.42
Output dim: 0, lower bound: -0.5224687, upper bound: 0.5419270
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -0.5824606, upper bound: 0.5340679
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -0.5737050, upper bound: 0.5410553
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -0.5551714, upper bound: 0.5731104
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.42
Output dim: 0, lower bound: -0.5224687, upper bound: 0.5650934

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.27 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5634311, upper bound: 0.5365798
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5535169, upper bound: 0.5509407
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.32 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5339610, upper bound: 0.5590787
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203890, upper bound: 0.5677562
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.27 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5291521, upper bound: 0.5672077
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203789, upper bound: 0.5770248
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.26 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5337286, upper bound: 0.5507160
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5203878, upper bound: 0.5634436
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5331724, upper bound: 0.5633422
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203876, upper bound: 0.5712291
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.31 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5262855, upper bound: 0.5673497
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203719, upper bound: 0.5771027
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.33 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5373765, upper bound: 0.5542694
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5228855, upper bound: 0.5654198
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5654198, upper bound: 0.5228855
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5542694, upper bound: 0.5373765
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.33 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5771027, upper bound: 0.5203719
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5673497, upper bound: 0.5262855
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 5

Time for candidate selection: 1.32 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5712291, upper bound: 0.5203876
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5633422, upper bound: 0.5331724
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 23

Time for candidate selection: 1.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5634436, upper bound: 0.5203878
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5507160, upper bound: 0.5337286
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 23

Time for candidate selection: 1.28 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5770248, upper bound: 0.5203789
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5672077, upper bound: 0.5291521
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 5

Time for candidate selection: 1.31 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5677562, upper bound: 0.5203890
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5590787, upper bound: 0.5339610
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.27 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5509407, upper bound: 0.5535169
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5365798, upper bound: 0.5634311
time: 0.36 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.58 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.5634311, upper bound: 0.5365798
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.5535169, upper bound: 0.5509407
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.5339610, upper bound: 0.5590787
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.5203890, upper bound: 0.5677562
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.5291521, upper bound: 0.5672077
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.5203789, upper bound: 0.5770248
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.5337286, upper bound: 0.5507160
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.5203878, upper bound: 0.5634436
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.5331724, upper bound: 0.5633422
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.5203876, upper bound: 0.5712291
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.5262855, upper bound: 0.5673497
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.5203719, upper bound: 0.5771027
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.5373765, upper bound: 0.5542694
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.5228855, upper bound: 0.5654198
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.5654198, upper bound: 0.5228855
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.5542694, upper bound: 0.5373765
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.5771027, upper bound: 0.5203719
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.5673497, upper bound: 0.5262855
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.5712291, upper bound: 0.5203876
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.5633422, upper bound: 0.5331724
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.5634436, upper bound: 0.5203878
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.5507160, upper bound: 0.5337286
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.5770248, upper bound: 0.5203789
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.5672077, upper bound: 0.5291521
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.5677562, upper bound: 0.5203890
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.5590787, upper bound: 0.5339610
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.5509407, upper bound: 0.5535169
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -0.5365798, upper bound: 0.5634311

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5203640, upper bound: 0.5213139
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203276, upper bound: 0.5675479
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5291295, upper bound: 0.5224842
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5291006, upper bound: 0.5670215
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5203539, upper bound: 0.5256562
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203224, upper bound: 0.5768506
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.61 seconds
Binary search (step 6): status=Status.UNKNOWN, low=0.1792442, high=0.1805312, mid=0.1805312, abs_max=0.6789970397949219
rel_dist={0: [-0.5996332167113175, 0.5996332167113172]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.17924420934288676
execution time: 1152.97 seconds
